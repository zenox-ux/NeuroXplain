import logging
import time
from collections import namedtuple
import time
import torch
from torch import nn

from protopnet.helpers import init_or_update
from protopnet.losses import WeightedIncorrectClassPrototypeActivations

logger = logging.getLogger(__name__)


# used to track where the prototypical part came from
prototype_meta = namedtuple("prototype_meta", ["sample_id", "sample_hash"])


class ProtoPNet(nn.Module):
    def __init__(
        self,
        backbone,
        add_on_layers,
        activation,
        prototype_layer,
        prototype_prediction_head,
        warn_on_errors: bool = False,
        k_for_topk: int = 1,
    ):
        super(ProtoPNet, self).__init__()

        self.backbone = backbone
        self.add_on_layers = add_on_layers
        self.activation = activation
        self.prototype_layer = prototype_layer
        self.prototype_prediction_head = prototype_prediction_head
        self.k_for_topk = k_for_topk

        self.__validate_model(warn_on_errors)

    def __validate_model(self, warn_on_errors: bool = False):
        """
        Validate the integretity of the model - namely, that the three layers are compatible with each other.
        """

        errors = []

        self.prototype_layer.latent_channels

        """
        if hasattr(self.add_on_layers, "proto_channels"):
            addon_latent_channels = self.add_on_layers.proto_channels
            if addon_latent_channels != prototype_layer_latent_channels:
                errors.append(
                    f"Backbone latent dimension {addon_latent_channels} does not match prototype layer latent dimension {prototype_layer_latent_channels}"
                )
        """

        if getattr(self.prototype_layer, "update_prototypes_on_batch", None) is None:
            errors.append(
                "Prototype layer does not have a push method. This is required for."
            )

        if len(errors) == 0:
            logger.debug("Model validation passed.")
        elif warn_on_errors:
            for error in errors:
                logger.warning(error)
        else:
            for error in errors:
                logger.error(error)
            raise ValueError(
                f"Model validation failed with {len(errors)}. See log for details."
            )

    def get_prototype_complexity(self, decimal_precision=8):
        """
        Computes and returns metrics about how many unique prototypes,
        unique parts, etc the model has
        Args:
            decimal_precision: The number of decimal places up to which we consider for
                equality. I.e., if decimal_precision = 8, 1e-9 equals 2e-9, but 1e-7 != 2e-7
        """
        return self.prototype_layer.get_prototype_complexity(
            decimal_precision=decimal_precision
        )

    def forward(
        self,
        x: torch.Tensor,
        sample_ids: list,
        return_prototype_layer_output_dict: bool = False,
        **kwargs,
    ):
        latent_vectors = self.backbone(x)
        latent_vectors = self.add_on_layers(latent_vectors)

        prototype_layer_output_dict = self.prototype_layer(latent_vectors, sample_ids)

        prototype_similarities = prototype_layer_output_dict["prototype_activations"]
        upsampled_activation = prototype_layer_output_dict["upsampled_activation"]

        prediction_logits = self.prototype_prediction_head(
            prototype_similarities, upsampled_activation, **kwargs
        )

        if return_prototype_layer_output_dict:
            output_dict = prediction_logits.copy()
            output_dict.update(prototype_layer_output_dict.copy())
            output_dict["latent_vectors"] = latent_vectors
            return output_dict
        else:
            return prediction_logits

    def prune_duplicate_prototypes(self, decimal_precision=8) -> None:
        assert (
            type(self.prototype_prediction_head) is PrototypePredictionHead
        ), "Error: Pruning only supports linear last layer at the moment"

        visited_unique_prototypes = None
        visited_prototype_class_identities = None
        visited_prototype_last_layer_weight = None

        update_proto_dict = len(self.prototype_layer.prototype_info_dict) > 0
        updated_prototype_info_dict = {}

        new_ind_for_proto = 0
        for proto_ind in range(self.prototype_tensors().shape[0]):
            cur_proto = self.prototype_tensors()[proto_ind].unsqueeze(0)
            if visited_unique_prototypes is None:
                visited_unique_prototypes = cur_proto
                visited_prototype_class_identities = (
                    self.prototype_layer.prototype_class_identity[proto_ind].unsqueeze(
                        0
                    )
                )
                visited_prototype_last_layer_weight = (
                    self.prototype_prediction_head.class_connection_layer.weight.data[
                        :, proto_ind
                    ].unsqueeze(1)
                )

                if update_proto_dict:
                    updated_prototype_info_dict[new_ind_for_proto] = (
                        self.prototype_layer.prototype_info_dict[proto_ind]
                    )
                    new_ind_for_proto += 1
            else:
                equiv_protos = (
                    torch.isclose(visited_unique_prototypes, cur_proto)
                    .all(axis=1)
                    .all(axis=1)
                    .all(axis=1)
                )
                if equiv_protos.any():
                    target_equiv_proto = torch.argmax(equiv_protos * 1)
                    visited_prototype_last_layer_weight[
                        :, target_equiv_proto
                    ] += self.prototype_prediction_head.class_connection_layer.weight.data[
                        :, proto_ind
                    ]
                else:
                    visited_unique_prototypes = torch.cat(
                        [visited_unique_prototypes, cur_proto], dim=0
                    )
                    visited_prototype_class_identities = torch.cat(
                        [
                            visited_prototype_class_identities,
                            self.prototype_layer.prototype_class_identity[
                                proto_ind
                            ].unsqueeze(0),
                        ],
                        dim=0,
                    )
                    visited_prototype_last_layer_weight = torch.cat(
                        [
                            visited_prototype_last_layer_weight,
                            self.prototype_prediction_head.class_connection_layer.weight.data[
                                :, proto_ind
                            ].unsqueeze(
                                1
                            ),
                        ],
                        dim=1,
                    )

                    if update_proto_dict:
                        updated_prototype_info_dict[new_ind_for_proto] = (
                            self.prototype_layer.prototype_info_dict[proto_ind]
                        )
                        new_ind_for_proto += 1

        logger.info(
            f"Pruning from {self.prototype_tensors().shape[0]} prototypes to {visited_unique_prototypes.shape[0]}"
        )
        self.prototype_layer.prototype_tensors = torch.nn.Parameter(
            visited_unique_prototypes
        )
        self.prototype_layer.prototype_class_identity = (
            visited_prototype_class_identities
        )
        new_last_layer = torch.nn.Linear(
            visited_unique_prototypes.shape[0],
            self.prototype_layer.num_classes,
            bias=False,
        ).to(self.prototype_layer.prototype_tensors.device)
        new_last_layer.weight.data.copy_(visited_prototype_last_layer_weight)
        self.prototype_prediction_head.class_connection_layer = new_last_layer
        self.prototype_layer.num_prototypes = visited_unique_prototypes.shape[0]

        if update_proto_dict:
            self.prototype_layer.prototype_info_dict = updated_prototype_info_dict

    def project(
        self, dataloader: torch.utils.data.DataLoader, class_specific=True
    ) -> None:
        logger.info("projecting prototypes onto %s", dataloader)
        state_before_push = self.training
        self.eval()
        start = time.time()

        # TODO: RENAME THIS
        n_prototypes = self.prototype_layer.num_prototypes

        global_max_proto_act = torch.full((n_prototypes,), -float("inf"))
        global_max_fmap_patches = torch.zeros_like(
            self.prototype_layer.prototype_tensors
        )

        search_batch_size = dataloader.batch_size

        logger.debug("initiating project batches")

        for push_iter, batch_data_dict in enumerate(dataloader):
            # TODO: ADD TQDM OPTIONALITY TO THIS LOOP
            logger.debug("starting project batch")
            search_batch_input = batch_data_dict["img"]
            search_y = batch_data_dict["target"]
            try:
                sample_ids = batch_data_dict["sample_id"]
            except KeyError:
                sample_ids = None

            start_index_of_search_batch = push_iter * search_batch_size

            search_batch_input = search_batch_input.to(
                self.prototype_layer.prototype_tensors.device
            )

            logger.debug("updating current best prototypes")
            self.prototype_layer.update_prototypes_on_batch(
                self.add_on_layers(self.backbone(search_batch_input)),
                start_index_of_search_batch,
                global_max_proto_act,
                global_max_fmap_patches,
                sample_ids,
                search_y,
                class_specific,
            )

            logger.debug("project batch complete")

        # set proto tensors
        self.prototype_layer.set_prototype_tensors(global_max_fmap_patches)

        """
        # if you're using dtw dictionary
        if hasattr(self.prototype_layer, "proto_dtw_dict"):
            self.prototype_layer.proto_dtw_dict.clear()
            
            # iter the info dict
            for key in self.prototype_layer.prototype_info_dict.keys():
                
                
                # grab the file name and path
                file_name = self.prototype_layer.prototype_info_dict[key]
                file_path = f"./DTW/final_dtw_v2/{file_name[0]}.pkl"
                
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)

                # update the dict to contain the newly loaded data
                self.prototype_layer.proto_dtw_dict.update(loaded_data)
        """

        end = time.time()
        logger.info("\tpush time: \t{0}".format(end - start))
        self.train(state_before_push)

    def prototype_tensors(self) -> torch.Tensor:
        return self.prototype_layer.prototype_tensors.data

    def get_prototype_class_identity(self, label) -> torch.Tensor:
        return self.prototype_layer.prototype_class_identity[:, label]


class LinearBatchLoss(nn.Module):
    def __init__(self, batch_losses: list = [], device="cpu"):
        super(LinearBatchLoss, self).__init__()
        self.batch_losses = batch_losses
        self.device = device

    def required_forward_results(self):
        return {
            req
            for loss_component in self.batch_losses
            for req in loss_component.loss.required_forward_results
        }

    def forward(self, **kwargs):
        # Metrics dict comes from kwargs
        metrics_dict = kwargs.get("metrics_dict", {})

        # TODO: Set device to be same as model based variables
        total_loss = torch.tensor(0.0, device=self.device)

        for loss_component in self.batch_losses:
            # Get args for loss from just the loss_component.required_forward_results from kwargs
            current_loss_args = {
                req: kwargs[req] for req in loss_component.loss.required_forward_results
            }

            # assert loss_component is a float
            current_loss = (
                loss_component.loss(**current_loss_args) * loss_component.coefficient
            )

            init_or_update(metrics_dict, loss_component.loss.name, current_loss.item())
            total_loss += current_loss

        return total_loss


class LinearModelRegularization(nn.Module):
    def __init__(self, model_losses: list = [], device="cpu"):
        super(LinearModelRegularization, self).__init__()
        self.model_losses = model_losses
        self.device = device

    def forward(self, model: ProtoPNet, **kwargs):
        metrics_dict = kwargs.get("metrics_dict", {})

        # TODO: Set device to be same as model based variables
        total_loss = torch.tensor(0.0, device=self.device)  # Adjust device as needed

        for loss_component in self.model_losses:
            current_loss = loss_component.loss(model) * loss_component.coefficient
            metrics_dict[loss_component.loss.name] = current_loss.item()
            total_loss += current_loss

        return total_loss


class ProtoPNetLoss(nn.Module):
    def __init__(self, batch_losses, model_losses, device="cpu"):
        super(ProtoPNetLoss, self).__init__()

        self.batch_loss = LinearBatchLoss(batch_losses, device)
        self.model_regularization = LinearModelRegularization(model_losses, device)

        self.batch_loss_required_forward_results = (
            self.batch_loss.required_forward_results()
        )

        self.incorrect_class_prototype_activations_fn = (
            WeightedIncorrectClassPrototypeActivations()
        )

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        model: ProtoPNet,
        metrics_dict: dict,
        **kwargs,
    ):
        # TODO: Make sure grad is being calculated here if grad_req

        # TODO: Is there a better way to do this syntax?
        if (
            "prototypes_of_correct_class" in self.batch_loss_required_forward_results
            or "prototypes_of_wrong_class" in self.batch_loss_required_forward_results
        ):
            prototypes_of_correct_class = torch.t(
                model.get_prototype_class_identity(target)
            )

            prototypes_of_wrong_class = 1 - prototypes_of_correct_class

        else:
            prototypes_of_correct_class = None
            prototypes_of_wrong_class = None

        if (
            "incorrect_class_prototype_activations"
            in self.batch_loss_required_forward_results
        ):
            # Fail fast- should not occur
            if (
                "similarity_score_to_each_prototype"
                not in self.batch_loss_required_forward_results
            ):
                raise ValueError(
                    "similarity_score_to_each_prototype is required for incorrect_class_prototype_activations"
                )
            else:
                similarity_score_to_each_prototype = kwargs[
                    "similarity_score_to_each_prototype"
                ]

            incorrect_class_prototype_activations = self.incorrect_class_prototype_activations_fn(
                similarity_score_to_each_prototype=similarity_score_to_each_prototype,
                prototypes_of_wrong_class=prototypes_of_wrong_class,
                target=target,
                prototype_class_identity=model.prototype_layer.prototype_class_identity,
            )

        if "prototype_class_identity" in self.batch_loss_required_forward_results:
            prototype_class_identity = model.prototype_layer.prototype_class_identity
        else:
            prototype_class_identity = None

        if "proto_batch_dtw_tensor" in self.batch_loss_required_forward_results:
            proto_dict = model.prototype_layer.prototype_info_dict
            kwargs["proto_dict"] = proto_dict

            if len(proto_dict) != 0:

                dtw_dict = model.prototype_layer.proto_dtw_dict
                num_protos = len(kwargs["proto_dict"])
                bsz_size = kwargs["nonweighted_simscores"].shape[0]

                all_dtw_tensor = torch.zeros(bsz_size, num_protos, 37)
                sample_index = 0

                for sample_id in kwargs[
                    "sample_ids"
                ]:  # for 1 input sample, get all prototes that correspond to it
                    input_eeg_name = sample_id

                    temp_list = []
                    for i in range(len(proto_dict)):  # iters the dict dimension

                        proto_eeg = proto_dict[i][0]  # [0] gets actual name
                        dtw_scores = dtw_dict[proto_eeg][
                            input_eeg_name
                        ]  # we FIX the sample, but iter the protos. Given tensor([37]) in output
                        temp_list.append(dtw_scores)

                    # stack all protos for the 1 input, shape (# proto, 37)
                    torch_dtw_scores = torch.stack(temp_list)
                    all_dtw_tensor[sample_index, :, :] = torch_dtw_scores
                    sample_index += 1  # move onto next input

                kwargs["proto_batch_dtw_tensor"] = all_dtw_tensor

            else:
                kwargs["all_dtw_tensor"] = None
                kwargs["proto_dict"] = proto_dict

        if "batch_pairwise_dtw_tensor" in self.batch_loss_required_forward_results:

            dtw_dict = model.prototype_layer.proto_dtw_dict
            num_protos = len(dtw_dict.keys())
            bsz_size = len(kwargs["sample_ids"])

            # ones to ensure that identity is same
            batch_pairwise_dtw_tensor = torch.ones(num_protos, bsz_size, 37)

            for sample_id_one in range(len(kwargs["sample_ids"])):
                for sample_id_two in range(
                    sample_id_one + 1, len(kwargs["sample_ids"])
                ):

                    # DTW[A,A] = 1! so the this skips it
                    eeg1 = kwargs["sample_ids"][sample_id_one]
                    eeg2 = kwargs["sample_ids"][sample_id_two]

                    batch_pairwise_dtw_tensor[sample_id_one, sample_id_two, :] = (
                        dtw_dict[eeg1][eeg2]
                    )
                    batch_pairwise_dtw_tensor[sample_id_two, sample_id_one, :] = (
                        dtw_dict[eeg1][eeg2]
                    )

            kwargs["batch_pairwise_dtw_tensor"] = batch_pairwise_dtw_tensor

        kwargs["spikenet_channel_dict"] = model.prototype_layer.spikenet_weight_dict

        # Pass in all arguments to batch_loss
        batch_loss = self.batch_loss(
            # pred=logits,
            target=target,
            fine_annotation=fine_annotation,
            # similarity_score_to_each_prototype=similarity_score_to_each_prototype,
            # upsampled_activation=upsampled_activation,
            prototype_class_identity=prototype_class_identity,
            prototypes_of_correct_class=prototypes_of_correct_class,
            prototypes_of_wrong_class=prototypes_of_wrong_class,
            incorrect_class_prototype_activations=incorrect_class_prototype_activations,
            metrics_dict=metrics_dict,
            **kwargs,
        )

        model_regularization = self.model_regularization(
            model, metrics_dict=metrics_dict
        )

        return batch_loss + model_regularization


class EmbeddedBackbone(nn.Module):
    """
    This is as backbone adapter for the original ProtoPNet implementation. It is used to wrap the backbone
    model and provide a common interface for the ProtoPNet architecture.
    """

    def __init__(self, embedded_model, input_channels: int = (3, 224, 224)):
        super(EmbeddedBackbone, self).__init__()
        self.embedded_model = embedded_model
        self.input_channels = input_channels

        self.latent_dimension = self.__latent_dimension()

    def forward(self, x: torch.Tensor):
        # Define the forward pass for the backbone
        return self.embedded_model(x)

    def __latent_dimension(self):
        """
        The latent dimension for each input (without the batch dimension). For example, if the backbone is a ResNet-18,
            then the latent dimension would be (512, 7, 7).

        Returns: latent_dimension (tuple): The latent dimension for each input (without the batch dimension).
        """
        dummy_tensor = torch.randn(1, *self.input_channels)
        return self.embedded_model(dummy_tensor).shape[1:]

    def __repr__(self):
        return f"EmbeddedBackbone({self.embedded_model})"


class AddonLayers(nn.Module):
    """
    This is an implementation of the optional add-on layers for a ProtoPNet, which lies between the
    backbone and the prototype prediction head
    """

    def __init__(
        self,
        num_prototypes: torch.Tensor,
        input_channels: int = 512,
        proto_channel_multiplier: float = 2**-2,
        num_addon_layers: int = 2,
    ):
        super(AddonLayers, self).__init__()

        self.num_prototypes = num_prototypes

        self.input_channels = input_channels

        self.proto_channels = int(proto_channel_multiplier * input_channels)

        if num_addon_layers == 0:
            if proto_channel_multiplier != 0:
                logger.warning(
                    f"""
                Proto channel multiplier is {proto_channel_multiplier}, but there are 0 addon layers. Ignoring multiplier
                """
                )
            self.add_on_layers = nn.Identity()
            self.proto_channels = input_channels
        else:
            mid_layers = []
            for _ in range(num_addon_layers - 1):
                mid_layers = mid_layers + [
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=self.proto_channels,
                        out_channels=self.proto_channels,
                        kernel_size=1,
                    ),
                ]
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.input_channels,
                    out_channels=self.proto_channels,
                    kernel_size=1,
                ),
                *mid_layers,
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor):
        # Define the forward pass for the backbone
        return self.add_on_layers(x)


class VanillaProtoPNet(ProtoPNet):
    def __init__(
        self,
        backbone: EmbeddedBackbone,
        add_on_layers,
        activation,
        num_classes: int,
        num_prototypes_per_class: int,
        k_for_topk: int = 1,
        **kwargs,
    ):
        num_prototypes = num_classes * num_prototypes_per_class

        # TODO: SHOULD BE CALLED FROM SAME INFO AS SELF.PROTOTYPE_INFO_DICT
        prototype_class_identity = torch.zeros(num_prototypes, num_classes)

        for j in range(num_prototypes):
            prototype_class_identity[j, j // num_prototypes_per_class] = 1

        prototype_config = {
            "num_classes": num_classes,
            "prototype_class_identity": prototype_class_identity,
            "k_for_topk": k_for_topk,
        }

        latent_channels = add_on_layers.proto_channels

        prototype_layer = PrototypeLayer(
            activation_function=activation,
            latent_channels=latent_channels,
            **prototype_config,
        )

        prediction_head = PrototypePredictionHead(**prototype_config)

        super(VanillaProtoPNet, self).__init__(
            backbone=backbone,
            add_on_layers=add_on_layers,
            activation=activation,
            prototype_layer=prototype_layer,
            prototype_prediction_head=prediction_head,
            k_for_topk=k_for_topk,
            **kwargs,
        )


# This skeleton defines the basic structure of classes as per the UML diagram.
# The actual implementation details like the model architecture, loss functions, and training procedures
# would need to be fleshed out based on the specific requirements of the ProtoPNet model.
