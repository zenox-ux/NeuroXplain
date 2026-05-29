import collections
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
import torchmetrics

from protopnet.helpers import get_learning_rates, init_or_update, predicated_extend, submit_full_pacmap_run
from protopnet.losses import (  # OffsetL2Cost,;
    ClusterCost,
    CrossEntropyCost,
    FineAnnotationCost,
    L1CostClassConnectionLayer,
    LossTerm,
    StackedOrthogonalityLoss,
    SeparationCost,
    ImportanceStatsReg
)
from protopnet.metrics import (
    PartConsistencyScore,
    PartStabilityScore,
    add_gaussian_noise,
)
from protopnet.skeleton import ProtoPNet, ProtoPNetLoss

log = logging.getLogger(__name__)


class ProtoPNetBackpropEpoch:
    def __init__(
        self,
        phase,
        train_backbone,
        train_add_on_layers,
        train_prototype_layer,
        train_conv_offset,
        train_prototype_prediction_head,
    ):
        self.phase = phase
        self.train_backbone = train_backbone
        self.train_add_on_layers = train_add_on_layers
        self.train_prototype_layer = train_prototype_layer
        self.train_prototype_prediction_head = train_prototype_prediction_head

        # TODO: Should we have a Deformable ProtoPNetEpoch?
        # To handle the case where we want to train the offset layer
        self.train_conv_offset = train_conv_offset

    def training_layers(self):
        # Features -> backbone
        # Prototype -> prototype layer

        # Add on layers
        # Conv offset
        return {
            "backbone": self.train_backbone,
            "add_on_layers": self.train_add_on_layers,
            "prototype_layer": self.train_prototype_layer,
            "conv_offset": self.train_conv_offset,
            "head": self.train_prototype_prediction_head,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(phase={self.phase})"


# Empty class for ProtoPNet Project Epoch
class ProtoPNetProjectEpoch:
    def __init__(self):
        self.phase = "project"

    def training_layers(self):
        return {
            "backbone": False,
            "add_on_layers": False,
            "prototype": True,
            "conv_offset": False,
            "head": False,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(phase={self.phase})"


def prototype_embedded_epoch(
    epoch: Union[ProtoPNetBackpropEpoch, ProtoPNetProjectEpoch]
):
    """
    Determines whether prototypes will match their embedprototype_embedded_epochding images after an epoch with the given settings.
    This is only True if the epoch is Project epoch, or the training does not affect the embedding.

    Returns:
        bool: True if prototypes will match their embedding images, False otherwise.
    """

    if isinstance(epoch, ProtoPNetProjectEpoch):
        return True

    layers = epoch.training_layers().copy()

    del layers["head"]

    return not any(layers.values())


class TrainingSchedule:
    def __init__(
        self,
        max_epochs=3000,
        num_warm_epochs=0,
        num_last_only_epochs=0,
        num_warm_pre_offset_epochs=0,
        num_joint_epochs=0,
        last_layer_fixed=False,
        project_epochs=[],
        num_last_only_epochs_after_each_project=20,
    ):
        # Check that num_last_only_epochs_after_each_project and num_last_only_epochs = 0
        if last_layer_fixed:
            assert (
                num_last_only_epochs == 0
                and num_last_only_epochs_after_each_project == 0
            ), "Cannot have last only epochs if last layer is fixed"

        self.max_epochs = max_epochs
        self.train_prototype_prediction_head = not last_layer_fixed
        self.num_last_only_epochs_after_each_project = (
            num_last_only_epochs_after_each_project
        )
        self.project_epochs = (
            self._convert_project_epochs_to_include_project_and_last_only_epochs(
                project_epochs
            )
        )

        self.epochs = self.build_vanilla_protopnet_training_schedule(
            num_warm_epochs,
            num_last_only_epochs,
            num_warm_pre_offset_epochs,
            num_joint_epochs,
            last_layer_fixed,
            self.project_epochs,
        )

    def _convert_project_epochs_to_include_project_and_last_only_epochs(
        self, project_epochs
    ):
        # Initialize a list to store the updated project epochs
        updated_project_epochs = []
        epochs_added_by_project_and_post_project_lastlayer_optimization = 0

        for project_epoch in project_epochs:
            updated_project_epochs.append(
                project_epoch
                + epochs_added_by_project_and_post_project_lastlayer_optimization
            )
            epochs_added_by_project_and_post_project_lastlayer_optimization += (
                1 + self.num_last_only_epochs_after_each_project
            )

        return updated_project_epochs

    def check_project_epoch_validity(self, project_epochs, base_training_epochs):
        if not project_epochs:
            return

        # Initialize a flag to indicate if the project epochs are valid
        are_project_epochs_valid = True

        for i, project_epoch in enumerate(project_epochs):
            # Calculate total epochs required up to this project epoch, considering last-only epochs after each project
            total_required_epochs = base_training_epochs + (i - 1) * (
                self.num_last_only_epochs_after_each_project + 1
            )

            # Check if the current project epoch is valid within the sequential timeline
            if project_epoch > total_required_epochs:
                are_project_epochs_valid = False
                break

        assert (
            are_project_epochs_valid
        ), "Project epochs must fit within the structured epoch timeline without creating a gap."

        # Ensure that project epochs are non-negative
        assert min(project_epochs) >= 0, "Project epochs must be non-negative. "

        # If project epoch is in project epochs at epoch i, then there should not be another project epoch for another num_last_only_epochs_after_each_project epochs
        sorted_project_epochs = sorted(project_epochs)
        assert all(
            [
                sorted_project_epochs[i] + self.num_last_only_epochs_after_each_project
                < sorted_project_epochs[i + 1]
                for i in range(len(sorted_project_epochs) - 1)
            ]
        ), "Project epochs must be separated by num_last_only_epochs_after_each_project epochs. "

    def build_vanilla_protopnet_training_schedule(
        self,
        num_warm_epochs,
        num_last_only_epochs,
        num_warm_pre_offset_epochs,
        num_joint_epochs,
        last_layer_fixed,
        project_epochs=[],
    ):
        assert any(
            [
                num_warm_epochs,
                num_last_only_epochs,
                num_warm_pre_offset_epochs,
                num_joint_epochs,
                project_epochs,
            ]
        ), "At least one of the epochs must be greater than 0 to train"

        assert (
            self.train_prototype_prediction_head or num_last_only_epochs == 0
        ), "Cannot have last only epochs if last layer is fixed"

        # num_last_only_epochs_after_each_project = 10
        # num_last_only_epochs = len(project_epochs) * num_last_only_epochs_after_each_project

        base_training_epochs = (
            num_warm_epochs
            + num_last_only_epochs
            + num_warm_pre_offset_epochs
            + num_joint_epochs
        )

        total_training_epochs = base_training_epochs + len(project_epochs)

        if not last_layer_fixed:
            total_training_epochs = (
                total_training_epochs
                + len(project_epochs) * self.num_last_only_epochs_after_each_project
            )

        self.check_project_epoch_validity(project_epochs, total_training_epochs)

        current_epoch = 0
        # TODO: Seems as though this logic could be removed if we just didn't count project epochs as epochs here
        # Can still count them for visualization/logging purposes too
        epochs_added_by_project_and_post_project_lastlayer_optimization = 0
        schedule = []

        while current_epoch < total_training_epochs:
            if current_epoch in project_epochs:
                schedule.append(ProtoPNetProjectEpoch())

                for i in range(self.num_last_only_epochs_after_each_project):
                    schedule.append(self._create_epoch("last_only"))
                    current_epoch += 1
                    epochs_added_by_project_and_post_project_lastlayer_optimization += 1

                # No increment to current_epoch because project epochs are not counted
                current_epoch += 1
                epochs_added_by_project_and_post_project_lastlayer_optimization += 1
                continue

            # Determine the phase based on the current_epoch
            if (
                current_epoch
                - epochs_added_by_project_and_post_project_lastlayer_optimization
                < num_warm_epochs
            ):
                schedule.append(self._create_epoch("warm"))
            # TODO: Is this necessary/right? Don't have a precedent for where to put this...
            elif (
                current_epoch
                - epochs_added_by_project_and_post_project_lastlayer_optimization
                < num_warm_epochs + num_last_only_epochs
            ):
                schedule.append(self._create_epoch("last_only"))
            elif (
                current_epoch
                - epochs_added_by_project_and_post_project_lastlayer_optimization
                < num_warm_epochs + num_last_only_epochs + num_warm_pre_offset_epochs
            ):
                schedule.append(self._create_epoch("warm_pre_offset"))
            else:
                schedule.append(self._create_epoch("joint"))

            current_epoch += 1  # Move to the next epoch considering all types

        # TODO: What to do if after num epochs
        # for epoch in project_epochs:
        #     if epoch >= total_training_epochs:
        #         schedule.append(ProtoPNetProjectEpoch())

        return schedule[: self.max_epochs]

    def _create_epoch(self, phase):
        # This helper function returns the appropriate epoch configuration based on the phase
        if phase == "warm":
            return ProtoPNetBackpropEpoch(
                phase=phase,
                train_backbone=False,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_conv_offset=False,
                train_prototype_prediction_head=self.train_prototype_prediction_head,
            )
        elif phase == "last_only":
            return ProtoPNetBackpropEpoch(
                phase=phase,
                train_backbone=False,
                train_add_on_layers=False,
                train_prototype_layer=False,
                train_conv_offset=False,
                train_prototype_prediction_head=self.train_prototype_prediction_head,
            )
        elif phase == "warm_pre_offset":
            return ProtoPNetBackpropEpoch(
                phase=phase,
                train_backbone=True,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_conv_offset=False,
                train_prototype_prediction_head=self.train_prototype_prediction_head,
            )
        elif phase == "joint":
            return ProtoPNetBackpropEpoch(
                phase=phase,
                train_backbone=True,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_conv_offset=True,
                train_prototype_prediction_head=False,
            )
        else:
            raise ValueError(f"Unsupported phase: {phase}")

    def get_epochs(self):
        return self.epochs

    # Naive __repr__ implementation that lists every single epoch
    def __repr_long__(self):
        schedule_repr = ",\n    ".join(repr(epoch) for epoch in self.epochs)
        return (
            f"{self.__class__.__name__}(max_epochs={self.max_epochs}, "
            f"train_prototype_prediction_head={self.train_prototype_prediction_head}, "
            f"epochs=[\n    {schedule_repr}\n])"
        )

    # __repr implementation that lists ranges of epochs
    def __repr__(self):
        """
        Returns a string representation of the TrainingSchedule object, summarizing the training epochs and their phases.

        The method groups consecutive epochs with the same phase together and displays them as ranges, providing a concise overview of the training plan. Phases for backprop epochs include 'warm', 'warm_pre_offset', 'last_only', and 'joint'. All project epochs have the phase 'project'. If the schedule is empty, it returns a placeholder string indicating an empty training schedule.

        Example Outputs:
            - If the schedule consists of 20 'warm' epochs followed by 10 'last_only' epochs, and then 5 'project' epochs, the output will be:
                TrainingSchedule(max_epochs=35, train_prototype_prediction_head=False, phases=[
                    1-20: ProtoPNetBackpropEpoch(phase=warm),
                    21-30: ProtoPNetBackpropEpoch(phase=last_only),
                    31-35: ProtoPNetProjectEpoch(phase=project)
                ])

            - If the schedule is empty, the output will be:
                <Empty TrainingSchedule>

        Returns:
            str: A string representation of the TrainingSchedule object.

        """

        phase_ranges = []
        if not self.epochs:
            return "<Empty TrainingSchedule>"

        # Initialize with the first epoch's phase
        current_phase = repr(self.epochs[0])
        start_epoch = 1
        handled_last_epoch = False

        for i, epoch in enumerate(self.epochs[1:], start=2):
            if repr(epoch) != current_phase or i == len(self.epochs):
                # Determine the end epoch for the current phase range
                end_epoch = (
                    i
                    if i == len(self.epochs) and repr(epoch) == current_phase
                    else i - 1
                )

                # Append the current phase range
                if start_epoch == end_epoch:
                    phase_ranges.append(f"{start_epoch}: {current_phase}")
                else:
                    phase_ranges.append(f"{start_epoch}-{end_epoch}: {current_phase}")

                if i == len(self.epochs) and repr(epoch) == current_phase:
                    handled_last_epoch = True  # The last epoch has been included

                current_phase = repr(epoch)
                start_epoch = i

        # Append the last phase range if it hasn't been handled
        if not handled_last_epoch:
            end_epoch = len(self.epochs)
            if start_epoch == end_epoch:
                phase_ranges.append(f"{start_epoch}: {current_phase}")
            else:
                phase_ranges.append(f"{start_epoch}-{end_epoch}: {current_phase}")

        return (
            f"{self.__class__.__name__}(max_epochs={self.max_epochs}, "
            f"train_prototype_prediction_head={self.train_prototype_prediction_head}, "
            f"phases=[\n    " + ",\n    ".join(phase_ranges) + "\n])"
        )


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    mode: str
    metric_source: collections.abc.Callable
    after_project: bool = True
    best: float = None
    counter: int = 0
    stop: bool = False

    def __post_init__(self):
        self.mode = self.mode.lower()
        assert self.mode in ["min", "max"], "mode must be 'min' or 'max'"
        self.best = float("inf") if self.mode == "min" else float("-inf")

    def check(self):
        value = self.metric_source()
        if self.mode == "min":
            if value < self.best - self.min_delta:
                self.best = value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if value > self.best + self.min_delta:
                self.best = value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop

    def reset(self):
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.stop = False

    def __repr__(self):
        return f"{self.__class__.__name__}(patience={self.patience}, min_delta={self.min_delta}, mode={self.mode}, monitor={self.monitor}, best={self.best}, counter={self.counter}, stop={self.stop})"

    def __str__(self):
        return f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, mode={self.mode}, monitor={self.monitor}, best={self.best}, counter={self.counter}, stop={self.stop})"


@dataclass
class TrainingMetric:
    name: str
    # min, max
    metric: torchmetrics.Metric

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, metric={self.metric})"

    def __str__(self):
        return f"{self.name}: {self.metric.compute()}"


# TODO: Make all the metrics be calculated in the same way
class TrainingMetrics:
    def __init__(
        self,
        metrics: List[TrainingMetric] = [],
        device: Union[str, torch.device] = "cpu",
    ):
        self.metrics = {metric.name: metric for metric in metrics}

        for metric in self.metrics.values():
            metric.metric.to(device)

    def start_epoch(self, phase: str):
        self.reset()

    def end_epoch(self, phase: str):
        self.reset()

    def metric_names(self):
        return list(self.metrics.keys())

    def reset(self):
        for _, metric in self.metrics.items():
            metric.metric.reset()

    def compute_dict(self) -> dict:
        """
        Compute all the metrics and return the raw values in a dictionary.
        """
        return {
            metric_name: metric.metric.compute()
            for metric_name, metric in self.metrics.items()
        }

    def update_all(self, forward_args: dict, forward_outputs: dict, phase: str):
        if len(self.metrics) > 0:
            raise NotImplementedError("This method must be implemented in a subclass.")

        # otherwise, do nothing, which is fine


class InterpretableTrainingMetrics(TrainingMetrics):
    """
    This is a temporary implementation of training metrics that lets us easily aggregate the interpretable
    metrics during an initial training run.
    """

    def __init__(
        self,
        protopnet: ProtoPNet,
        num_classes: int,
        proto_per_class: int,
        part_num: int,
        img_size: 224,
        half_size: 36,
        device: Union[str, torch.device] = "cpu",
        acc_only: bool = False,
    ):
        """
        Args:
            protopnet (ProtoPNet): The ProtoPNet model.
            num_classes (int): The number of classes in the dataset.
            proto_per_class (int): The default number of prototypes per class.
            part_num (int): The number of parts in the dataset.
            img_size (int): The size of the input images.
            half_size (int): The size of the half of the input image. See: metrics.InterpretableMetrics
            device (Union[str, torch.device], optional): The device to run the metrics on. Defaults to "cpu".
        """

        super().__init__(
            metrics=predicated_extend(
                not acc_only,
                [
                    TrainingMetric(
                        name="accuracy",
                        metric=torchmetrics.Accuracy(
                            num_classes=num_classes,
                            task="multiclass",
                        ),
                    ),
                ],
                [
                    TrainingMetric(
                        name="prototype_stability",
                        metric=PartStabilityScore(
                            num_classes=num_classes,
                            part_num=part_num,
                            proto_per_class=proto_per_class,
                            img_sz=img_size,
                            half_size=half_size,
                        ),
                    ),
                    TrainingMetric(
                        name="prototype_consistency",
                        metric=PartConsistencyScore(
                            num_classes=num_classes,
                            part_num=part_num,
                            proto_per_class=proto_per_class,
                            img_sz=img_size,
                            half_size=half_size,
                        ),
                    ),
                    TrainingMetric(
                        name="prototype_sparsity",
                        metric=torchmetrics.MeanMetric(),
                    ),
                    TrainingMetric(
                        name="n_unique_proto_parts",
                        metric=torchmetrics.MeanMetric(),
                    ),
                    TrainingMetric(
                        name="n_unique_protos",
                        metric=torchmetrics.MeanMetric(),
                    ),
                ],
            ),
            device=device,
        )

        self.generator = torch.Generator(device=device)
        self.protopnet = protopnet

        self.prototype_metrics_cached = False
        # FIXME - this is hack
        self.acc_only = acc_only

    def metric_names(self):
        raw_metric_names = super().metric_names()
        if self.acc_only:
            return raw_metric_names
        else:
            return raw_metric_names + [
                "prototype_score",
                "acc_proto_score",
            ]

    def start_epoch(self, phase: str):
        if phase and phase == "project":
            self.reset()
            self.prototype_metrics_cached = False
        else:
            self.metrics["accuracy"].metric.reset()

    def update_all(self, forward_args: dict, forward_outputs: dict, phase: str):
        """
        Update all the metrics.
        """
        self.update_accuracy(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )

        if phase == "project" and not self.acc_only:
            self.update_stability(
                forward_args=forward_args,
                forward_outputs=forward_outputs,
            )
            self.update_consistency(
                forward_args=forward_args,
                forward_outputs=forward_outputs,
            )
            self.update_prototype_sparsity()

    def update_accuracy(self, forward_args: dict, forward_outputs: dict):
        """
        Update the accuracy metric.
        """
        accuracy = self.metrics["accuracy"].metric
        accuracy.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_stability(self, forward_args: dict, forward_outputs: dict):
        """
        Update the stability metric.
        """
        with torch.no_grad():
            # Compute the consistency metric (prototype_activations_noisy)
            proto_acts_noisy = self.protopnet(
                add_gaussian_noise(forward_args["img"], self.generator),
                return_prototype_layer_output_dict=True,
            )["prototype_activations"].detach()

        stability = self.metrics["prototype_stability"].metric
        stability.update(
            proto_acts=forward_outputs["prototype_activations"],
            targets=forward_args["target"],
            proto_acts_noisy=proto_acts_noisy,
            sample_parts_centroids=forward_args["sample_parts_centroids"],
            sample_bounding_box=forward_args["sample_bounding_box"],
        )

    def update_consistency(self, forward_args: dict, forward_outputs: dict):
        """
        Update the consistency metric.
        """
        consistency = self.metrics["prototype_consistency"].metric

        consistency.update(
            proto_acts=forward_outputs["prototype_activations"],
            targets=forward_args["target"],
            sample_parts_centroids=forward_args["sample_parts_centroids"],
            sample_bounding_box=forward_args["sample_bounding_box"],
        )

    def update_prototype_sparsity(self):
        """
        Update the prototype sparsity metric.
        """
        prototype_complexity_stats = self.protopnet.get_prototype_complexity()

        self.metrics["prototype_sparsity"].metric.update(
            prototype_complexity_stats["prototype_sparsity"]
        )
        self.metrics["n_unique_protos"].metric.update(
            prototype_complexity_stats["n_unique_protos"]
        )
        self.metrics["n_unique_proto_parts"].metric.update(
            prototype_complexity_stats["n_unique_proto_parts"]
        )

    def compute_dict(self) -> dict:
        """
        Compute all the metrics and return the raw values in a dictionary.
        """
        if self.acc_only:
            return {"accuracy": self.metrics["accuracy"].metric.compute()}

        if self.prototype_metrics_cached:
            log.debug("returning cached metrics")
            result_dict = self.cached_results
            result_dict["accuracy"] = self.metrics["accuracy"].metric.compute()

        else:
            log.debug("calculating new metrics")
            result_dict = {
                metric_name: metric.metric.compute()
                for metric_name, metric in self.metrics.items()
            }

            result_dict["prototype_score"] = (
                min(result_dict["prototype_sparsity"], 1.0)
                + result_dict["prototype_stability"]
                + result_dict["prototype_consistency"]
            ) / 3

            self.prototype_metrics_cached = True
            self.cached_results = result_dict.copy()

        result_dict["acc_proto_score"] = (
            result_dict["prototype_score"] * result_dict["accuracy"]
        )

        return result_dict


class ProtoPNetTrainer:
    def __init__(
        self,
        model,
        dataloader,
        activation_function,
        optimizers_with_schedulers,
        device,
        coefs=None,
        with_fa=False,
        fa_type="l2",
        use_ortho_loss=False,
        class_specific=True,
        deformable=False,
        project_dataloader=None,
        val_dataloader=None,
        logger=None,
        early_stopping_patience=None,
        save_dir=Path(os.environ.get("PPNXT_ARTIFACT_DIR", "models")),
        min_save_threshold=0.0,
        min_post_project_target_metric=0.0,
        num_accumulation_batches=1,
        training_metrics: TrainingMetrics = None,
        target_metric_name="accu",
        compute_metrics_for_embed_only=True,
        run_pacmap = False
    ):
        # Change to use a real logger (create_logger)

        # model, dataloader, activation_function, optimizer=None, device="cuda"
        self.model = model
        self.dataloader = dataloader
        self.activation_function = activation_function
        
        print("trainer 768 proto layer act: ", type(activation_function))

        # TODO: Add assert statement to ensure there is an optimizer for each phase
        self.optimizers_with_schedulers = optimizers_with_schedulers

        self.compute_metrics_for_embed_only = compute_metrics_for_embed_only
        self.device = device
        self.with_fa = with_fa
        self.fa_type = fa_type
        self.use_ortho_loss = use_ortho_loss
        self.class_specific = class_specific
        self.deformable = deformable
        self.coefs = coefs
        self.num_accumulation_batches = num_accumulation_batches
        self.run_pacmap = run_pacmap

        self.model = self.model.to(self.device)

        # Number of projects without improvement before stopping (0 means stop on first project without improvement over previous best)
        self.early_stopping_patience = early_stopping_patience
        self.min_post_project_target_metric = min_post_project_target_metric

        # TODO: Determine if this is where it should be set
        # But also want to allow for a Callable coefficient that would be function of the epoch
        batch_losses = [
            LossTerm(loss=CrossEntropyCost(), coefficient=self.coefs["cross_entropy"]),
            LossTerm(
                loss=ClusterCost(class_specific=self.class_specific),
                coefficient=self.coefs["cluster"],
            ),
            LossTerm(loss=SeparationCost(), coefficient=self.coefs["separation"]),
        ]

        if self.with_fa:
            batch_losses.append(
                LossTerm(
                    loss=FineAnnotationCost(fa_loss=self.fa_type),
                    coefficient=self.coefs["fa"],
                )
            )

        model_losses = [
            LossTerm(loss=L1CostClassConnectionLayer(), coefficient=self.coefs["l1"]),
            LossTerm(loss=ImportanceStatsReg(), coefficient=self.coefs["importance_stats"])
        ]

        if "orthogonality_loss" in self.coefs:
            # confirmed that this works
            model_losses.append(
                LossTerm(
                    loss=StackedOrthogonalityLoss(),
                    coefficient=self.coefs["orthogonality_loss"],
                )
            )

        # TODO: Determine a better way to update the device without passing device into the loss
        self.loss = ProtoPNetLoss(
            batch_losses=batch_losses, model_losses=model_losses, device=self.device
        )

        self.forward_calc_flags = {
            f"return_{req}": True
            for req in self.loss.batch_loss.required_forward_results()
        }

        # TODO: Consolidate this with above
        self.forward_calc_flags["return_prototype_layer_output_dict"] = True

        if logger is None:
            # TODO consolidate our logger initializations
            self.logger = TensorBoardLogger(
                use_ortho_loss=use_ortho_loss,
                class_specific=class_specific,
                device=self.device,
            )
        else:
            self.logger = logger
            self.logger.device = self.device

        # self.metric_logger = MetricLogger(device=self.device)

        self.project_dataloader = project_dataloader
        # TODO: check self.project_dataloader return dictionary has a string return for sample_id (only needs to be checked once)

        # TODO: Determine if this needs to be passed in here or if it can be passed in at a different time (seems inflexible rn)
        self.val_dataloader = val_dataloader

        self.save_dir = save_dir
        self.min_save_threshold = min_save_threshold
        self.target_metric_name = target_metric_name

        self.training_metrics = training_metrics

    def update_training_phase(self, epoch_settings):
        # Map model components to training settings dynamically
        for name, param in self.model.named_parameters():
            # Extract the component name from the parameter name
            # Assuming the naming convention follows the pattern "<component_name>_..."
            component_name = name.split(".")[0]  # Get the first part of the name

            # Construct the setting attribute name dynamically
            setting_attr = f"train_{component_name}"

            # Check if the corresponding setting attribute exists in epoch_settings
            # TODO: Pre-validate
            assert hasattr(
                epoch_settings, setting_attr
            ), f"Attribute '{setting_attr}' not found in epoch_settings"

            # Since the attribute exists, use getattr to fetch its value
            # The third argument in getattr is not needed anymore since we're asserting the attribute's existence
            should_train = getattr(epoch_settings, setting_attr)

            # Update the requires_grad based on the setting
            param.requires_grad = should_train

    def train(self, training_schedule, val_each_epoch=True, save_model=True):
        # TODO: Should this check be here?
        if val_each_epoch:
            assert (
                self.val_dataloader
            ), "val_dataloader must be provided if val_each_epoch is True"

        log.info("Training with the following schedule:")
        log.info("%s", repr(training_schedule))

        if save_model:
            assert val_each_epoch, "Must run evaluation epochs to save model"
            
            if self.run_pacmap:
                try:
                    append_name = os.environ["WANDB_RUN_ID"]
                except:
                    append_name = "test"

                self.save_dir = os.path.join(self.save_dir, append_name)
                
            os.makedirs(self.save_dir, exist_ok=True)

        # parsimonious early stopping
        last_eval_target_metric = float("-inf")
        best_preproject_target_metric = float("-inf")
        best_project_target_metric = float("-inf")
        early_stopping_project_count = 0

        for epoch_index, epoch_settings in enumerate(training_schedule.get_epochs()):
            
            if epoch_index == 0 and self.run_pacmap:
                submit_full_pacmap_run(self.dataloader, self.model, self.save_dir, "init" )
            
            # If epoch_settings of type ProtoPNetBackpropEpoch
            if isinstance(epoch_settings, ProtoPNetBackpropEpoch):
                log.info(
                    f"Starting Epoch {epoch_index} of Phase {epoch_settings.phase} with settings: {epoch_settings.training_layers()}"
                )
                self.update_training_phase(epoch_settings)
                self.train_epoch(
                    phase=epoch_settings.phase,
                    epoch_index=epoch_index,
                    epoch_settings=epoch_settings,
                )
            elif isinstance(epoch_settings, ProtoPNetProjectEpoch):
                # TODO: Make a fail fast that ensures that self.project_dataloader is not None (while only checking once)
                log.info(f"Starting Epoch {epoch_index} as Project Epoch")
                
                if self.run_pacmap and target_metric > 2.5:
                    submit_full_pacmap_run(self.dataloader, self.model, self.save_dir, f"{epoch_index}_prepush" )
                    
                self.project_epoch(epoch_index, epoch_settings)
                
                if self.run_pacmap and target_metric > 2.5:
                    submit_full_pacmap_run(self.dataloader, self.model, self.save_dir, f"{epoch_index}_postpush" )
                # TODO: Run self.eval_epoch() but with dataloader=self.dataloader
                # Want to evaluate on train

            # TODO: Make a fail-fast using assert statements
            else:
                raise ValueError(
                    f"Unsupported type of epoch_settings: {type(epoch_settings)}"
                )

            if val_each_epoch:
                log.info(f"Starting Validation Epoch {epoch_index}")
                target_metric = self.eval_epoch(epoch_index, epoch_settings)
            else:
                # TODO: smarter about determining accuracy without eval
                target_metric = float("-inf")

            # TODO: Undo hard-coding
            if save_model and prototype_embedded_epoch(epoch_settings):
                previous_best = self.logger.bests[self.target_metric_name][
                    "prototypes_embedded"
                ]
                # TODO: weird order - technically we just updated best
                if (
                    target_metric >= previous_best
                    and target_metric > self.min_save_threshold
                ):
                    model_name = f"{str(epoch_index)}_{epoch_settings.phase}"
                    metric_path = "_{0:.4f}.pth".format(float(target_metric))
                    model_path = os.path.join(self.save_dir, model_name + metric_path)
                    log.info(
                        "Saving model with %s %s to %s",
                        self.target_metric_name,
                        target_metric,
                        model_path,
                    )
                    torch.save(
                        obj=self.model,
                        f=model_path,
                    )
                else:
                    log.debug(
                        "skipping saving model state with %s %s",
                        self.target_metric_name,
                        target_metric,
                    )
            # parsimonious early stopping
            if isinstance(epoch_settings, ProtoPNetProjectEpoch):
                if self.early_stopping_patience is not None:
                    if (
                        last_eval_target_metric <= best_preproject_target_metric
                        and target_metric <= best_project_target_metric
                    ):
                        early_stopping_project_count += 1
                    else:
                        early_stopping_project_count = 0

                    if early_stopping_project_count > self.early_stopping_patience:
                        log.info("Early stopping after %s epochs", epoch_index + 1)
                        log.info(
                            "Best accuracy before project: %s",
                            best_preproject_target_metric,
                        )
                        log.info(
                            "Best accuracy after project: %s",
                            best_project_target_metric,
                        )
                        log.info(
                            "%s projects without improvement",
                            early_stopping_project_count,
                        )
                        break
                    else:
                        best_project_target_metric = max(
                            best_project_target_metric, target_metric
                        )
                        best_preproject_target_metric = max(
                            best_preproject_target_metric, last_eval_target_metric
                        )

                if (
                    self.min_post_project_target_metric is not None
                    and target_metric <= self.min_post_project_target_metric
                ):
                    log.info(
                        "Early stopping after %s epochs because post project threshold of %s not exceeded by %s",
                        epoch_index + 1,
                        self.min_post_project_target_metric,
                        target_metric,
                    )
                    break

            last_eval_target_metric = target_metric

        log.info("Training complete after %s epochs", epoch_index + 1)
        return epoch_index

    def project_epoch(self, epoch_index, epoch_settings):
        # TODO: Combine logic with train_epoch?

        start = time.time()
        self.model.eval()
        with torch.no_grad():
            self.model.project(self.project_dataloader)
        end = time.time()
        log.info(f"Completed project epoch in {end - start} seconds")

    def train_epoch(self, phase, epoch_index, epoch_settings):
        """
        Conducts a single epoch of training.
        """
        self.model.train()
        with torch.enable_grad():
            self.run_epoch(
                dataloader=self.dataloader,
                optimizer_scheduler=self.optimizers_with_schedulers[phase],
                epoch_index=epoch_index,
                epoch_settings=epoch_settings,
                compute_metrics_this_epoch=False,
            )

    def eval_epoch(self, epoch_index, epoch_settings):
        """
        Conducts a single epoch of testing/validation.
        """
        self.model.eval()
        with torch.no_grad():
            target_metric = self.run_epoch(
                dataloader=self.val_dataloader,
                epoch_index=epoch_index,
                epoch_settings=epoch_settings,
                compute_metrics_this_epoch=(
                    prototype_embedded_epoch(epoch_settings)
                    or not self.compute_metrics_for_embed_only
                ),
            )

        return target_metric

    def run_epoch(
        self,
        dataloader,
        epoch_index,
        epoch_settings,
        optimizer_scheduler=None,
        compute_metrics_this_epoch=False,
    ):
        # model,dataloader,activation_function,optimizer=None,class_specific=True,use_l1_mask=True,coefs=None,log=print,subtractive_margin=True,use_ortho_loss=False,finer_loader=None,fa_loss="serial",device="cuda",

        # Should be able to use functions rather than all these if statements
        # Create TrainAdapter, TestAdapter, and ValidationAdapter classes
        # Also create Adapters for Deformable, Fine Annotation, etc

        # TODO: Add a way to track variables as None/NA if they aren't used
        # Or make this more flexible to be more flexible in the terms it tracks
        epoch_metrics_dict = {
            "time": None,
            "n_examples": None,
            "n_correct": None,
            "n_batches": None,
            "cross_entropy": None,
            "mse": None,
            "cluster": None,
            "contrastive": None,
            "separation": None,
            "fine_annotation": None,
            "orthogonality": None,
            "accu": None,
            "l1": None,
            "l1_weightlayer": None,
            "importance_stats": None,
            "total_loss": None,
            "n_unique_proto_parts": None,
            "n_unique_protos": None,
            "prototype_non_sparsity": None,
        }

        # FIXME: there should always be metrics
        if self.training_metrics is not None and compute_metrics_this_epoch:
            self.training_metrics.start_epoch(phase=epoch_settings.phase)

        optimizer, scheduler = (
            optimizer_scheduler if optimizer_scheduler is not None else (None, None)
        )

        if optimizer is not None:
            lr_log = get_learning_rates(
                optimizer=optimizer, model=self.model, detailed=False
            )
            self.logger.log_backdrops(lr_log, step=epoch_index)

        start = time.time()

        # Use a helper function to handle None values

        if optimizer:
            optimizer.zero_grad()
        agg_loss = 0
        total_correct = 0
        total_samples = 0
        mse_arr = []
        predictions_array = []
        count = 0
        for i, batch_data_dict in enumerate(dataloader):
            count += 1
            # TODO: Make this formatting better

            # Intended to include sample IDs for metadata logging while also allowing for the case where the dataloader does not return sample IDs
            # Perhaps could make dataloader create a list of sample IDs if it does not already have one
            image = batch_data_dict["img"]
            label = batch_data_dict["target"]


            try: # this is only for protoeegnet
                sample_ids = batch_data_dict["sample_id"]
            except KeyError:
                pass


            if self.with_fa:
                fine_anno = batch_data_dict["fine_anno"]
                fine_annotation = fine_anno.to(self.device)
            else:
                fine_anno = None
                fine_annotation = None

            # TODO: Remove this ugly formatting
            input = image.to(self.device)
            target = label.to(self.device)

            # TODO: Subtractive margin
            output = self.model(input, sample_ids, **self.forward_calc_flags)   

            try: # this is only for protoeegnet
                output["sample_ids"] = sample_ids
            except:
                pass

            # conv_features = self.model.backbone(input)

            # TODO: Move to forward of Deformable Proto Layer
            with torch.no_grad():
                logits = output["logits"]


                # Binary predictions from sigmoid probability
                pred_probs = torch.sigmoid(logits[:, 0])
                predicted = (pred_probs >= 0.5).long()
                mse_loss = F.mse_loss(pred_probs, target.float()).item()

                batch_acc = (predicted == target).float().mean().item()


                init_or_update(epoch_metrics_dict, "n_examples", target.size(0))
                init_or_update(epoch_metrics_dict, "mse", mse_loss)
                init_or_update(
                    epoch_metrics_dict, "n_correct", (predicted == target).sum().item()
                )
                init_or_update(epoch_metrics_dict, "n_batches", 1)

            loss = (
                self.loss(
                    target=target,
                    model=self.model,
                    metrics_dict=epoch_metrics_dict,
                    fine_annotation=fine_annotation,
                    **output,
                )
                / self.num_accumulation_batches
            )

            agg_loss += loss.item()
            if optimizer:
                loss.backward(retain_graph=True)

            # Check if we have reached our accumulation threshold
            if ((i + 1) % self.num_accumulation_batches == 0) or (
                i + 1 == len(dataloader)
            ):
                init_or_update(epoch_metrics_dict, "total_loss", agg_loss)
                agg_loss = 0
                if optimizer:
                    # self.optimizer.step())
                    optimizer.step()
                    optimizer.zero_grad()

            # FIXME: There should always be metrics
            if self.training_metrics is not None and compute_metrics_this_epoch:
                log.debug("updating extra metrics")

                # FIXME: somewhere these tensors are being moved off of gpu
                for key, maybe_tensor in batch_data_dict.items():
                    if (
                        isinstance(maybe_tensor, torch.Tensor)
                        and maybe_tensor.device != self.device
                    ):
                        batch_data_dict[key] = maybe_tensor.to(self.device)
                for key, maybe_tensor in output.items():
                    if (
                        isinstance(maybe_tensor, torch.Tensor)
                        and maybe_tensor.device != self.device
                    ):
                        output[key] = maybe_tensor.to(self.device)

                with torch.no_grad():
                    self.training_metrics.update_all(
                        batch_data_dict, output, phase=epoch_settings.phase
                    )
                log.debug("Extra metrics updated")

            # Removed: offsets, batch_max, additional_returns
            del (
                input,
                target,
                fine_annotation,
                output,
                predicted,
            )  # similarity_score_to_each_prototype, batch_max

            del image, label, fine_anno

            del loss

        #print("total accu for this epoch: ", total_correct/total_samples)
        #print("averaged mse for this epoch: ", sum(mse_arr)/len(mse_arr))
        #print("predictions array: ", sum(predictions_array)/len(predictions_array))


        end = time.time()

        if scheduler:
            # Step scheduler if possible
            scheduler.step()

        epoch_metrics_dict["time"] = end - start

        # TODO: Determine where to make these calculations
        epoch_metrics_dict["accu"] = (
            100 * epoch_metrics_dict["n_correct"] / epoch_metrics_dict["n_examples"]
        )


        #print("in trainer 1268 accu: ", epoch_metrics_dict["accu"])

        if "offset_bias_l2" in self.coefs and "avg_l2" in epoch_metrics_dict:
            epoch_metrics_dict["avg_l2_with_weight"] = (
                self.coefs["offset_bias_l2"] * epoch_metrics_dict["avg_l2"]
            )

        if (
            "orthogonality_loss" in self.coefs
            and "orthogonality_loss" in epoch_metrics_dict
        ):
            epoch_metrics_dict["orthogonality"] = (
                self.coefs["orthogonality_loss"]
                * epoch_metrics_dict["orthogonality_loss"]
            )

        if self.training_metrics is not None and compute_metrics_this_epoch:
            log.debug("Computing extra metrics")
            start = time.time()
            with torch.no_grad():
                extra_training_metrics = self.training_metrics.compute_dict()
            log.info("Extra metrics calculated in %s", time.time() - start)
        else:
            extra_training_metrics = None

        # TODO: Assess if flags should be passed in here or into the init of the Logger class
        self.logger.end_epoch(
            epoch_metrics_dict,
            is_train=True if optimizer else False,
            epoch_index=epoch_index,
            prototype_embedded_epoch=prototype_embedded_epoch(epoch_settings),
            precalculated_metrics=extra_training_metrics,
        )

        # FIXME: There should always be metrics
        if self.training_metrics is not None:
            self.training_metrics.end_epoch(phase=epoch_settings.phase)

        # FIXME: Consolidate metrics
        if self.target_metric_name in epoch_metrics_dict:
            return epoch_metrics_dict[self.target_metric_name]
        elif (
            extra_training_metrics and self.target_metric_name in extra_training_metrics
        ):
            return extra_training_metrics[self.target_metric_name]
        else:
            # FIXME: this is hack for eval-only target metrics, but it shouldn't break anything
            return 0.0


class TrainLogger:
    def __init__(
        self,
        use_ortho_loss=False,
        class_specific=True,
        # FIXME: this should consistently be called accuracy
        calculate_best_for=["accu"],
        device="cpu",
    ):
        self.use_ortho_loss = use_ortho_loss
        self.class_specific = class_specific
        # self.coefs = coefs
        # FIXME: this should support min and max
        self.bests = self.__setup_bests(calculate_best_for)

        # Create separate metrics dictionaries for train and validation
        self.train_metrics = self.create_metrics(device)
        self.val_metrics = self.create_metrics(device)

    # FIXME: this should be part of the metrics class, not the logger
    def __setup_bests(self, calculate_best_for):
        bests = {}
        for metric_name in calculate_best_for:
            # FIXME: this should support min and max
            bests[metric_name] = {
                "any": float("-inf"),
                "prototypes_embedded": float("-inf"),
            }

        return bests

    # FIXME: this should be part of the metrics class, not the logger
    def update_bests(self, metrics_dict, step, prototype_embedded_epoch=False):
        for metric_name, metric_value in metrics_dict.items():
            if metric_name in self.bests and metric_value is not None:
                if metric_value > self.bests[metric_name]["any"]:
                    self.bests[metric_name]["any"] = metric_value
                    self.process_new_best(
                        self.__metric_best_name(metric_name, False), metric_value, step
                    )

                if prototype_embedded_epoch:
                    if metric_value > self.bests[metric_name]["prototypes_embedded"]:
                        self.bests[metric_name]["prototypes_embedded"] = metric_value
                        self.process_new_best(
                            self.__metric_best_name(metric_name, True),
                            metric_value,
                            step,
                        )

    def __metric_best_name(self, metric_name, prototype_embedded_state):
        # FIXME: we should consistently call this accuracy throughout
        maybe_prototypes_embedded = (
            "prototypes_embedded_" if prototype_embedded_state else ""
        )
        return f"best_{maybe_prototypes_embedded}{metric_name}"

    def serialize_bests(self):
        bests_flat = {}
        for metric_name, metric_values in self.bests.items():
            bests_flat[self.__metric_best_name(metric_name, False)] = metric_values[
                "any"
            ]
            bests_flat[self.__metric_best_name(metric_name, True)] = metric_values[
                "prototypes_embedded"
            ]
        return bests_flat

    def process_new_best(
        self, metric_name, metric_value, step, prototype_embedded_state=False
    ):
        """
        This method is called whenever a new "best" value of a metric is found with the value of the metric, the current, step,
        and whether the prototype layer is embedded or not. It provides a hook to capture the new value and take any necessary actions.

        The default is a no-op. Subclasses can override this method to implement custom behavior.
        """
        pass

    def create_metrics(self, device):
        # Helper method to initialize metrics
        return {
            "n_examples": torchmetrics.SumMetric().to(device),
            "n_correct": torchmetrics.SumMetric().to(device),
            "n_batches": torchmetrics.SumMetric().to(device),
            "cross_entropy": torchmetrics.MeanMetric().to(device),
            "contrastive": torchmetrics.MeanMetric().to(device),
            "mse": torchmetrics.MeanMetric().to(device),
            "cluster": torchmetrics.MeanMetric().to(device),
            "separation": torchmetrics.MeanMetric().to(device),
            "fine_annotation": torchmetrics.MeanMetric().to(device),
            "orthogonality": torchmetrics.MeanMetric().to(device),
            "accu": torchmetrics.MeanMetric().to(
                device
            ),  # Using torchmetrics.Accuracy directly for accuracy
            "l1": torchmetrics.SumMetric().to(device),
            "l1_weightlayer": torchmetrics.SumMetric().to(device),
            "importance_stats": torchmetrics.SumMetric().to(device),
            "total_loss": torchmetrics.MeanMetric().to(device),
        }

    def update_metrics(self, metrics_dict, is_train):
        metrics = self.train_metrics if is_train else self.val_metrics

        # Update each metric from metrics_dict
        for key, value in metrics_dict.items():
            # TODO: Is this desired- not tracking Nones (torchmetrics does not like them)
            if key in metrics and value is not None:
                metrics[key].update(value)


class TensorBoardLogger(TrainLogger):
    def __init__(
        self,
        use_ortho_loss=False,
        class_specific=True,
        device="cpu",
        calculate_best_for=["accu"],
    ):
        super().__init__(
            use_ortho_loss=use_ortho_loss,
            class_specific=class_specific,
            device=device,
            calculate_best_for=calculate_best_for,
        )

    def log_metrics(
        self,
        is_train,
        precalculated_metrics=None,
        prototype_embedded_state=False,
        step=None,
    ):
        metrics = self.train_metrics if is_train else self.val_metrics
        tag = "train" if is_train else "validation"

        # Log the computed metric values
        for name, metric in metrics.items():
            computed_value = metric.compute()
            log.info(f"{name} ({tag}): {computed_value}")
            metric.reset()  # Reset after logging for the next epoch

        # TODO: Unify with other metrics
        if precalculated_metrics:
            for name, value in precalculated_metrics.items():
                log.info(f"{name}: {value}")

    def end_epoch(
        self,
        epoch_metrics_dict,
        is_train,
        epoch_index,
        prototype_embedded_epoch,
        precalculated_metrics=None,
    ):
        if self.use_ortho_loss:
            log.info("\t Using ortho loss")

        for key in epoch_metrics_dict:
            # DO NOTHING FOR THESE KEYS
            if (
                key
                not in [
                    "time",
                    "n_batches",
                    "l1",
                    "l1_weightlayer",
                    "importance_stats",
                    "max_offset",
                    "n_correct",
                    "n_examples",
                    "accu",
                    "is_train",
                ]
                and epoch_metrics_dict[key]
            ):
                epoch_metrics_dict[key] /= epoch_metrics_dict["n_batches"]

        self.update_metrics(epoch_metrics_dict, is_train)

        complete_metrics = epoch_metrics_dict.copy()
        if precalculated_metrics is not None:
            complete_metrics.update(precalculated_metrics)

        self.update_bests(
            complete_metrics,
            step=epoch_index,
            prototype_embedded_epoch=prototype_embedded_epoch,
        )
        self.log_metrics(
            is_train,
            prototype_embedded_state=prototype_embedded_epoch,
            precalculated_metrics=precalculated_metrics,
            step=epoch_index,
        )

        for key in epoch_metrics_dict:
            # if class specific is true, print separation and avg_separation
            # always print the rest
            if self.class_specific or key not in ["separation", "avg_separation"]:
                log.info(
                    "\t{0}: \t{1}".format(key, epoch_metrics_dict[key]),
                )

    @staticmethod
    def log_backdrops(backdrop_dict, step=None):
        for name, value in backdrop_dict.items():
            log.info(f"{name}: {value}")