from collections import namedtuple
from typing import Callable
import random
import torch
import torch.nn.functional as F
from torch import nn

from protopnet.helpers import custom_unravel_index, hash_func


from .activations import CosPrototypeActivation

# used to track where the prototypical part came from
prototype_meta = namedtuple("prototype_meta", ["sample_id", "sample_hash"])


class PrototypeLayer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        activation_function: Callable,
        prototype_class_identity: torch.Tensor,
        latent_channels: int = 512,
        prototype_dimension: tuple = (1, 1),
        k_for_topk: int = 1,
        init_normal: bool = False,
    ):
        super(PrototypeLayer, self).__init__()
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.latent_channels = latent_channels
        self.latent_spatial_size = None

        # TODO: REVIEW THAT THIS CONTAINS DESIRED METADATA
        self.prototype_info_dict = dict()

        # self.proto_dtw_dict = dict()
        self.proto_dtw_dict = None

        self.with_fa = False

        self.num_prototypes = prototype_class_identity.shape[0]

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes

        # TODO: Determine if this is the correct procedure for avoiding device problems (prototype_class_identity on cpu)
        # Could also just explicitly set it to the device when we set the model
        self.register_buffer("prototype_class_identity", prototype_class_identity)

        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        if init_normal:
            check = True
            self.prototype_tensors = nn.Parameter(
                torch.randn(
                    self.num_prototypes,
                    latent_channels,
                    *prototype_dimension,
                    requires_grad=True,
                )
            )

        else:
            self.prototype_tensors = nn.Parameter(
                torch.rand(
                    self.num_prototypes,
                    latent_channels,
                    *prototype_dimension,
                    requires_grad=True,
                )
            )

        # delete this later
        assert check == True
        self.k_for_topk = k_for_topk

    def get_prototype_complexity(self, decimal_precision=8):
        """
        Computes and returns metrics about how many unique prototypes,
        unique parts, etc the model has
        Args:
            decimal_precision: The number of decimal places up to which we consider for
                equality. I.e., if decimal_precision = 8, 1e-9 equals 2e-9, but 1e-7 != 2e-7
        """
        # Reorganize so that we have a collection of prototype part vectors
        part_vectors = self.prototype_tensors.permute(0, 2, 3, 1).reshape(
            -1, self.prototype_tensors.shape[1]
        )
        n_unique_proto_parts = (
            torch.round(part_vectors, decimals=decimal_precision).unique(dim=0).shape[0]
        )

        # Repeat to get the number of unique prototype tensors
        stacked_proto_vectors = self.prototype_tensors.reshape(
            self.prototype_tensors.shape[0], -1
        )
        n_unique_protos = (
            torch.round(stacked_proto_vectors, decimals=decimal_precision)
            .unique(dim=0)
            .shape[0]
        )

        min_sparsity = self.num_classes * (
            1 + 1 / (self.latent_spatial_size[0] * self.latent_spatial_size[1])
        )
        prototype_sparsity = n_unique_protos + n_unique_proto_parts / (
            self.latent_spatial_size[0] * self.latent_spatial_size[1]
        )

        prototype_sparsity = min_sparsity / prototype_sparsity

        return {
            "n_unique_proto_parts": n_unique_proto_parts,
            "n_unique_protos": n_unique_protos,
            "prototype_sparsity": prototype_sparsity,
        }

    def forward(self, x: torch.Tensor):
        """
        Provides a prototype similarity for each image at each location. This results in a tensor of shape
        (batch_size, num_prototypes, latent_height, latent_width)
        """
        # x - [90, 128, 1, 37]
        # self.prototype_tensor - [45, 128, 1, 1] 45 = # protos
        # prototype_activations - [90, 45, 1, 37]
        prototype_activations = self.activation_function(x, self.prototype_tensors)

        if not hasattr(self, "latent_spatial_size") or self.latent_spatial_size is None:
            self.latent_spatial_size = (
                prototype_activations.shape[-2],
                prototype_activations.shape[-1],
            )

        # TODO: Add upsampled activation
        if self.with_fa:
            upsampled_activation = torch.nn.Upsample(
                size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
            )(prototype_activations)
        else:
            upsampled_activation = None

        output_dict = {
            "prototype_activations": prototype_activations,
            "upsampled_activation": upsampled_activation,
        }

        return output_dict

    def set_prototype_tensors(self, new_prototype_tensors):
        prototype_update = torch.reshape(
            new_prototype_tensors,
            tuple(self.prototype_tensors.shape),
        )

        self.prototype_tensors.data.copy_(
            torch.tensor(prototype_update, dtype=torch.float32).to(
                self.prototype_tensors.device
            )
        )

    def update_prototypes_on_batch(
        self,
        protoL_input_torch,
        start_index_of_search_batch,
        global_max_proto_act,
        global_max_fmap_patches,
        sample_ids,
        search_y,
        class_specific,
    ):
        # TODO: DESIGN DECISIONS TO BE MADE
        # TODO: CLASS SPECIFIC = TRUE IS NOT PROPERLY IMPLEMENTED
        prototype_layer_stride = 1

        # Assuming data is on correct device; setup belongs in the trainer
        # TODO: ALL ON CUDA OR NOT
        proto_act_torch = self.forward(
            protoL_input_torch.to(self.prototype_tensors.device), sample_ids
        )["prototype_activations"]

        # protoL_input_ = torch.clone(protoL_input_torch.detach().cpu())
        # proto_act_ = torch.clone(proto_act_torch.detach().cpu())

        # del protoL_input_torch, proto_act_torch

        if class_specific:
            # Index class_to_img_index dict with class number, return list of images
            class_to_img_index_dict = {key: [] for key in range(self.num_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(search_y):
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)

        prototype_shape = self.prototype_tensors.shape

        for j in range(self.num_prototypes):
            class_index = j

            if class_specific:
                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(
                    self.prototype_class_identity[class_index]
                ).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_act_j = proto_act_torch[class_to_img_index_dict[target_class]][
                    :, j, :, :
                ]
            else:
                # if it is not class specific, then we will search through
                # every example
                proto_act_j = proto_act_torch[:, j, :, :]

            batch_max_proto_act_j = torch.amax(proto_act_j)

            if batch_max_proto_act_j > global_max_proto_act[j]:
                batch_argmax_proto_act_j = list(
                    custom_unravel_index(
                        torch.argmax(proto_act_j, axis=None), proto_act_j.shape
                    )
                )
                if class_specific:
                    """
                    change the argmin index from the index among
                    images of the target class to the index in the entire search
                    batch
                    """
                    batch_argmax_proto_act_j[0] = class_to_img_index_dict[target_class][
                        batch_argmax_proto_act_j[0]
                    ]

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmax_proto_act_j[0]
                fmap_height_start_index = (
                    batch_argmax_proto_act_j[1] * prototype_layer_stride
                )
                fmap_width_start_index = (
                    batch_argmax_proto_act_j[2] * prototype_layer_stride
                )

                # TODO: REVISIT SHAPE INDEXING
                fmap_height_end_index = fmap_height_start_index + prototype_shape[-2]
                fmap_width_end_index = fmap_width_start_index + prototype_shape[-1]

                batch_max_fmap_patch_j = protoL_input_torch[
                    img_index_in_batch,
                    :,
                    fmap_height_start_index:fmap_height_end_index,
                    fmap_width_start_index:fmap_width_end_index,
                ]

                # TODO: CONSTRUCT DICTIONARY OUTSIDE THE LOOP ONCE
                if sample_ids is not None:
                    self.prototype_info_dict[j] = prototype_meta(
                        sample_ids[img_index_in_batch],
                        hash_func(protoL_input_torch[img_index_in_batch]),
                    )

                global_max_proto_act[j] = batch_max_proto_act_j
                global_max_fmap_patches[j] = batch_max_fmap_patch_j



class WeightedPrototypeLayer(PrototypeLayer):

    def __init__(
        self,
        num_classes: int = 2,
        activation_function: int = 2,
        prototype_class_identity: torch.Tensor = None,
        latent_channels: int = 512,
        prototype_dimension: tuple = (1, 37),  # change the default change
        k_for_topk: int = 1,
        init_normal: bool = False,
        push_data=None,
        backbone=None,
        use_handcrafted_features=True,
        use_spikenet_channelweights=True,
    ):

        super(WeightedPrototypeLayer, self).__init__(
            num_classes,
            activation_function,
            prototype_class_identity,
            latent_channels,
            prototype_dimension,
            k_for_topk,
            init_normal=True,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        backbone = backbone.to(device)

        self.sftmx = torch.nn.Softmax(dim=0)

        ########## init the prototype tensors + init ##########
        if use_spikenet_channelweights == True:
            self.spikenet_weight_dict = torch.load("model_feats/spikenet_labels.pth")

        else:
            self.spikenet_weight_dict = torch.load(
                "model_feats/channelwise_identity.pth"
            )

        empty_proto_tensor = torch.zeros(
            (self.num_prototypes, latent_channels, *prototype_dimension)
        )

        # just some sanity checks
        for i in range(empty_proto_tensor.shape[0]):
            assert empty_proto_tensor[i].mean() == 0

        protos_per_class = int(self.num_prototypes / num_classes)
        count_dict = {i: 0 for i in range(num_classes)}

        torch.rand(self.num_prototypes, *prototype_dimension)

        # count dict counts the number of placed protos for each class
        while not all(value == protos_per_class for value in count_dict.values()):
            # Select random input
            random_int = random.randint(0, len(push_data) - 1)
            input = push_data[random_int]["img"]
            label = push_data[random_int]["target"]
            push_data[random_int]["sample_id"]

            # continue if reached max for a specific class
            if count_dict[label] == protos_per_class:
                continue

            # assuming that you ARE adding to the prototype tensor
            output = backbone(input.unsqueeze(0).to(device))

            # actually set the empty proto tensor| count_dict[label] is how have already been place
            place_index = label * protos_per_class + count_dict[label]
            empty_proto_tensor[place_index] = output

            # increase the placement count by 1
            count_dict[label] += 1

            # uncomment for manifold init
            # self.prototype_info_dict[place_index] = prototype_meta(sample_id, 0)

        # assert all columns are filled
        # for i in range(empty_proto_tensor.shape[0]):
        #     assert empty_proto_tensor[i].mean() != 0

        # uncomment for manifold init
        # self.prototype_tensors = nn.Parameter(empty_proto_tensor, requires_grad = True)

        if use_handcrafted_features:
            # order is: latent|range|var|fft
            self.importance_by_statistic = nn.Parameter(
                torch.log(
                    torch.tensor([0.5, 0.125, 0.0625, 0.3125], requires_grad=False)
                )
            )

        else:
            self.importance_by_statistic = torch.tensor(
                [999999, 0, 0, 0], dtype=torch.float32
            )

    def forward(self, x: torch.Tensor, sample_ids: list):

        # input is shape [90, 258, 1, 37]
        init_weight_tensor = torch.zeros((x.shape[0], 1, 37))

        #
        count = 0
        for sample_id in sample_ids:

            if sample_id in self.spikenet_weight_dict:
                init_weight_tensor[count] = self.spikenet_weight_dict[sample_id] / (
                    self.spikenet_weight_dict[sample_id].sum() + 0.000001
                )
            else:
                init_weight_tensor[count] = torch.ones(1, 37) / 37  # uniform fallback
            count += 1

        # masked_weight_tensor = (init_weight_tensor > 0.015).float()
        masked_weight_tensor = init_weight_tensor

        softmaxxed_importance_by_statistic = self.sftmx(self.importance_by_statistic)

        output_dict = self.activation_function(
            x,
            prototype_tensors=self.prototype_tensors,
            weight_tensors=masked_weight_tensor,
            importance_by_statistic=softmaxxed_importance_by_statistic,
            proto_class_identity=self.prototype_class_identity,
        )

        output_dict["upsampled_activation"] = None

        return output_dict
