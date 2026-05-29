import logging
import sys
import warnings
from torch import nn
import torch

import wandb
import numpy as np

from .spikenet_helpers import BalancedBatchSampler

from .backbones import construct_backbone
from .skeleton import (
    AddonLayers,
    ProtoPNet,
)
from .activations import WeightedCosineSimilarityWithStats
from .prototype_layers import WeightedPrototypeLayer
from .prediction_heads import PrototypeBinaryClassificationPredictionHead


from .trainer import ProtoPNetTrainer, TrainingSchedule
from .weights_and_biases import WeightsAndBiasesTrainLogger

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def run(
    *,
    backbone="spikenet_summary",
    pre_project_phase_len=1,  # was 25
    num_warm_pre_offset_epochs=0,
    phase_multiplier=1,  # for online augmentation
    latent_dim_exp=7,  # 128
    joint_lr_step_size=1,  # was 10
    post_project_phases=4,  # was 4
    joint_epochs_per_phase=18,
    last_only_epochs_per_phase=0,
    num_prototypes_per_class=10,
    cluster_coef=0.00,
    importance_stats=1.0,
    separation_coef=0,
    l1_coef=0.00,  # was 0
    contrastive=10,
    l1_weightlayer_coef=1,
    latent_dim_multiplier_exp=-2,
    num_addon_layers=0,
    with_fa=False,
    fa_type=None,
    fa_coef=0,
    offset_weight_l2=0,
    orthogonality_loss=0.00,  # was 0
    offset_bias_l2=0,
    cross_entropy=2,
    protoweight_alignment=1,
    k_for_topk=1,
    joint_add_on_lr_multiplier=1,  # was 1
    warm_lr_multiplier=1,  # was 1
    lr_multiplier=1.4507750558924228,  # was 1
    last_layer_lr_multiplier=0.24625457807672912,  # was 1
    prototype_dimension=(1, 37),
    bias_value=3,  # was -1
    gamma=0.5,  # was 0.5
    warm_init=True,
    dry_run=False,
    verify=False,
    interpretable_metrics=False,
    bias_lr=0.05,
    seed=0,
    use_handcrafted_features=True,
    use_spikenet_channelweights=True,
):
    """
    Train a ProtoEEGNet.

    Args:
    - backbone: str - See backbones.py
    - pre_project_phase_len: int - number of epochs in each pre-project phase (warm-up, joint). Total preproject epochs is 2*pre_project_phase_len*phase_multiplier.
    - phase_multiplier: int - for each phase, multiply the number of epochs in that phase by this number
    - latent_dim_exp: int - expotential of 2 for the latent dimension of the prototype layer
    - joint_lr_step_size: int - number of epochs between each step in the joint learning rate scheduler. Multiplied by phase_multiplier.
    - last_only_epochs_per_phase: int - coefficient for clustering loss
    - post_project_phases: int - number of times to iterate between last-only, joint, project after the initial pre-project phases
    - cluster_coef: float - coefficient for clustering loss term
    - separation_coef: float - coefficient for separation loss term
    - l1_coef: float - coefficient for clustering loss
    - fa_type: str - one of "serial", "l2", or "square" to indicate which type of fine annotation loss to use. if None, fine annotation is deactivated.
    - fa_coef: float - coefficient for fine annotation loss term
    - num_prototypes_per_class: int - number of prototypes to create for each class
    - lr_multiplier: float - multiplier for learning rates. The base values are from protopnet's training.
    - dry_run: bool - Configure the training run, but do not execute it
    - preflight: bool - Configure a training run for a single epoch of all phases
    """
    set_seed(seed)
    backbone_name = backbone

    # TODO: this should be controlled elsewhere
    if wandb.run is None:
        wandb.init(
            entity="faiqqazi73-nust",
            project="PROTOEEG",
            
        )

    if verify:
        logger.info("Setting preflight configuration to all 1s")
        pre_project_phase_len = 1
        post_project_phases = 1
        joint_epochs_per_phase = 1
        last_only_epochs_per_phase = 1
        phase_multiplier = 1
        num_prototypes_per_class = 1
        num_prototypes_per_class = 16

        prototype_dimension = (3, 3)

    if fa_type is not None and fa_coef == 0:
        warnings.warn("Run set up to use Fine Annotations, but fa_coef set to 0.")
    elif fa_type is None and fa_coef != 0:
        warnings.warn(
            f"Run set up to not use Fine Annotations, but fa_coef set to {fa_coef}."
        )

    setup = {
        # FIXME: remove after merge
        "batch_size": 8,
        "num_classes": 2,
        "coefs": {
            "cluster": cluster_coef,
            "offset_weight_l2": offset_weight_l2,
            "separation": separation_coef,
            "orthogonality_loss": orthogonality_loss,
            "offset_bias_l2": offset_bias_l2,
            "importance_stats": importance_stats,
            "l1": l1_coef,
            "fa": fa_coef,
            "contrastive": contrastive,
            "cross_entropy": cross_entropy,
            "protoweight_alignment": protoweight_alignment,
            "l1_weightlayer": l1_weightlayer_coef,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "train_log_filename": "train_log.txt",
    }

    num_accumulation_batches = 1
    batch_sizes = {"train": 8, "push": 8, "val": 8}

    if type(prototype_dimension) is int:
        prototype_dimension = (prototype_dimension, prototype_dimension)

    # TODO: Share this logic with ProtoPNet
    num_warm_epochs = pre_project_phase_len * phase_multiplier
    # accounting for warm and joint
    total_pre_project = 2 * (pre_project_phase_len) * phase_multiplier

    true_last_only_epochs_per_phase = last_only_epochs_per_phase * phase_multiplier
    num_post_project_backprop_epochs = (
        post_project_phases
        * (joint_epochs_per_phase + last_only_epochs_per_phase)
        * phase_multiplier
    )
    num_joint_epochs = joint_epochs_per_phase * post_project_phases * phase_multiplier
    # NOTE: the last-only epochs are added by the training schedule, so this schedule is just the joint between projects
    joint_between_project = joint_epochs_per_phase * phase_multiplier

    project_epochs = [
        e
        for e in range(
            total_pre_project,
            num_post_project_backprop_epochs + total_pre_project - 1,
            joint_between_project,
        )
    ]

    schedule = TrainingSchedule(
        num_warm_epochs=num_warm_epochs,
        num_last_only_epochs=0,
        num_warm_pre_offset_epochs=num_warm_pre_offset_epochs,
        num_joint_epochs=num_joint_epochs,
        max_epochs=num_post_project_backprop_epochs + total_pre_project,
        last_layer_fixed=False,
        project_epochs=project_epochs,
        num_last_only_epochs_after_each_project=true_last_only_epochs_per_phase,
    )

    fancy_activation_function = WeightedCosineSimilarityWithStats()

    num_prototypes = setup["num_classes"] * num_prototypes_per_class
    prototype_class_identity = torch.zeros(num_prototypes, setup["num_classes"])

    for j in range(num_prototypes):
        prototype_class_identity[j, j // num_prototypes_per_class] = 1

    backbone = construct_backbone(backbone_name)

    x = torch.ones((10, 1, 128, 37))

    ###############################################################################################################
    import copy

    # convert the (1,37) channel to be (1,1)
    conv = getattr(backbone.embedded_model.model, "8")
    new_weight = torch.mean(conv.weight, axis=-1).unsqueeze(-1)
    conv.kernel_size = (1, 1)
    conv.weight.data = new_weight
    setattr(backbone.embedded_model.model, "8", conv)
    # confirmed this averages the 37 dims, set the (8,1) to be (9,1)

    conv = getattr(backbone.embedded_model.model, "27")
    old_weight = copy.copy(conv.weight)
    old_mean = torch.mean(old_weight, axis=-2).unsqueeze(-1)
    new_weight = torch.cat((old_weight, old_mean), axis=-2)
    conv.kernel_size = (10, 1)
    conv.weight.data = new_weight

    # print("new weight shape: ", new_weight.shape)
    setattr(backbone.embedded_model.model, "27", conv)

    replace_list = ["4", "10", "14", "19", "23", "28"]
    size = [
        (32, 128, 37),
        (64, 32, 37),
        (64, 32, 37),
        (96, 8, 37),
        (96, 8, 37),
        (128, 1, 37),
    ]

    for i in range(len(replace_list)):
        # Get the number of features in the BatchNorm layer
        num_features = size[i]
        # Create a corresponding LayerNorm layer
        layer_norm = nn.LayerNorm(num_features).cuda()

        # Replace BatchNorm with LayerNorm
        setattr(backbone.embedded_model.model, replace_list[i], layer_norm)
    ##############################################################################################################

    backbone.cuda()
    input = torch.ones((10, 1, 128, 37)).float().cuda()

    prototype_config = {
        "k_for_topk": k_for_topk,
        "num_classes": setup["num_classes"],
        "prototype_class_identity": prototype_class_identity,
        "bias": bias_value,
    }
    prediction_head = PrototypeBinaryClassificationPredictionHead(**prototype_config)

    add_on_layers = AddonLayers(
        num_prototypes=num_prototypes_per_class * setup["num_classes"],
        input_channels=backbone.latent_dimension[0],
        proto_channel_multiplier=2**latent_dim_multiplier_exp,
        num_addon_layers=num_addon_layers,
    )

    if "summary" in backbone_name:
        add_on_layers.proto_channels = 258

        # dt161/TEEGLLTEEG/sn2_data/organized_data/sn2_train_labels.npy

    # train_loader, train_push_loader, val_loader = cub200.train_dataloaders(batch_sizes=batch_sizes)
    custom_dataset_name = "EEG_ConcatDataset"
    customDataSet_kw_args = {
        "eeg_data": {
            "train": "../sn2_data/organized_data/train_dict.pth",
            "train_push": "../sn2_data/organized_data/train_dict.pth",
            "eval": "../sn2_data/organized_data/val_dict.pth",
        },
        "labels": {
            "train": "../sn2_data/organized_data/sn2_train_labels.npy",
            "train_push": "../sn2_data/organized_data/sn2_train_labels.npy",
            "eval": "../sn2_data/organized_data/sn2_val_labels.npy",
        },
        "threshold": 0.5,
        "train_transform": None,
        "push_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover spikenet_helpers.normalizer",
        "eval_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover spikenet_helpers.normalizer",
    }

    _temp = __import__("eeg_utilities", globals(), locals(), ["custom_dataset"], 1)
    custom_dataset_module = getattr(_temp, "custom_dataset")
    custom_dataset_class = getattr(custom_dataset_module, custom_dataset_name)
    train_dataset = custom_dataset_class(mode="train", **customDataSet_kw_args)

    balanced_data_sampler = BalancedBatchSampler(train_dataset, 2, 8)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=balanced_data_sampler
    )

    train_push_dataset = custom_dataset_class(
        mode="train_push", **customDataSet_kw_args
    )

    push_loader_config = {"batch_size": 8, "shuffle": False, "pin_memory": False}

    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, **push_loader_config
    )

    val_dataset = custom_dataset_class(mode="eval", **customDataSet_kw_args)

    val_loader_config = {"batch_size": 8, "shuffle": False, "pin_memory": False}

    val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_config)

    # # SECTION FOR SHORT TRAINING
    # train_loader = torch.utils.data.Subset(train_dataset, range(0, 20))
    # train_loader = torch.utils.data.DataLoader(train_loader)
    # val_loader = torch.utils.data.Subset(val_dataset, range(1, 20))
    # val_loader = torch.utils.data.DataLoader(val_loader, **val_loader_config)
    # print("############ \n \n \n You are in test mode where data is short \n \n \n \n \n ###########")

    prototypes = WeightedPrototypeLayer(
        num_classes=setup["num_classes"],
        activation_function=fancy_activation_function,
        prototype_class_identity=prototype_class_identity,
        prototype_dimension=prototype_dimension,
        latent_channels=add_on_layers.proto_channels,  # was add_on_layers.proto_channels
        init_normal=True,
        push_data=train_push_dataset,
        backbone=backbone,
        use_handcrafted_features=use_handcrafted_features,
        use_spikenet_channelweights=use_spikenet_channelweights,
    )

    ##### CHECK WE WANT THE FANCY IN BOTHC ###

    ppn = ProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=fancy_activation_function,
        prototype_layer=prototypes,
        prototype_prediction_head=prediction_head,
    )

    ppn = ppn.to(setup["device"])

    warm_optimizer_lrs = {
        "prototype_tensors": 0.003 * warm_lr_multiplier * lr_multiplier,
        "add_on_layers": 0.00 * warm_lr_multiplier * lr_multiplier,
        # "importance_by_statistic": 0.001 * joint_add_on_lr_multiplier * lr_multiplier
    }

    warm_pre_offset_optimizer_lrs = {
        "joint_last_layer_lr": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "prototype_tensors": 0.003 * joint_add_on_lr_multiplier * lr_multiplier,
        "features": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "add_on_layers": 0.003 * joint_add_on_lr_multiplier * lr_multiplier,
    }

    joint_optimizer_lrs = {
        "joint_last_layer_lr": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "prototype_tensors": 0.01 * joint_add_on_lr_multiplier * lr_multiplier,
        "conv_offset": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "features": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "add_on_layers": 0.003 * joint_add_on_lr_multiplier * lr_multiplier,
        # "importance_by_statistic": 0.001 * joint_add_on_lr_multiplier * lr_multiplier
    }

    warm_optimizer_specs = [
        {
            "params": ppn.prototype_layer.prototype_tensors,
            "lr": warm_optimizer_lrs["prototype_tensors"],
        },
        # {
        #     #"params": ppn.prototype_layer.importance_by_statistic, # new
        #     #"lr": warm_optimizer_lrs["importance_by_statistic"],
        # },
    ]
    warm_pre_offset_optimizer_specs = [
        {
            "params": ppn.backbone.parameters(),
            "lr": warm_pre_offset_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },  # bias are now also being regularized
        {
            "params": ppn.prototype_layer.prototype_tensors,
            "lr": warm_pre_offset_optimizer_lrs["prototype_tensors"],
        },
        {
            "params": ppn.prototype_prediction_head.class_connection_layer.bias,
            "lr": bias_lr,
        },
    ]
    joint_optimizer_specs = [
        {
            "params": ppn.backbone.parameters(),
            "lr": joint_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },  # bias are now also being regularized0,
        {
            "params": ppn.prototype_layer.prototype_tensors,
            "lr": joint_optimizer_lrs["prototype_tensors"],
        },
        {
            "params": ppn.prototype_prediction_head.class_connection_layer.bias,
            "lr": bias_lr,
        },
        # {
        #     "params": ppn.prototype_layer.importance_by_statistic, # new
        #     "lr": joint_optimizer_lrs["importance_by_statistic"],
        # },
    ]

    last_layer_optimizer_specs = [
        {
            "params": ppn.prototype_prediction_head.class_connection_layer.bias,
            "lr": bias_lr,  # was 1e-3
        },
    ]

    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    warm_pre_offset_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs)
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # TODO: Make this step for each epoch
    # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     joint_optimizer, step_size=joint_lr_step_size, gamma=bias_lr
    # )

    optimizers_with_schedulers = {
        "warm": (warm_optimizer, None),  # No scheduler for warm-up phase
        "joint": (
            joint_optimizer,
            torch.optim.lr_scheduler.StepLR(
                joint_optimizer,
                step_size=joint_lr_step_size * phase_multiplier,
                gamma=0.9,
            ),
        ),
        # Add the joint LR scheduler
        "last_only": (
            last_layer_optimizer,
            None,
        ),
        "warm_pre_offset": (
            warm_pre_offset_optimizer,
            None,
        ),
    }

    train_logger = WeightsAndBiasesTrainLogger(device=setup["device"])

    ppn_trainer = ProtoPNetTrainer(
        model=ppn,
        dataloader=train_loader,
        activation_function=fancy_activation_function,
        optimizers_with_schedulers=optimizers_with_schedulers,
        device=setup["device"],
        coefs=setup["coefs"],
        class_specific=True,
        with_fa=fa_type is not None,
        fa_type=fa_type,
        project_dataloader=train_push_loader,
        val_dataloader=val_loader,
        early_stopping_patience=None,
        logger=train_logger,
        num_accumulation_batches=num_accumulation_batches,
        run_pacmap=False,
    )

    if dry_run:
        logger.info("Skipping training due to dry run: %s", schedule)
    else:
        ppn_trainer.train(schedule)