from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from protopnet.skeleton import ProtoPNet

from dataclasses import dataclass
from typing import Callable, Union
from protopnet.spikenet_helpers import threshold_temperature_softmax


@dataclass
class LossTerm:
    loss: nn.Module
    coefficient: Union[Callable, float]


class L1PrototypeWeightLayer(nn.Module):
    def __init__(self):
        super(L1PrototypeWeightLayer, self).__init__()
        self.name = "l1_weightlayer"

    def forward(self, model: "ProtoPNet"):
        # shape of model.prototype_layer.weight_tensors [num proto, 1, 37]

        # old l1 code
        # l1 = torch.linalg.matrix_norm(model.prototype_layer.weight_tensors, ord = 1)
        # mean_l1 = torch.mean(l1)

        k = 8
        top_k_values, _ = torch.topk(
            torch.abs(model.prototype_layer.weight_tensors).squeeze(1), k, dim=1
        )
        mean_top_k = torch.mean(top_k_values, dim=1)

        mean_values = torch.mean(
            torch.abs(model.prototype_layer.weight_tensors).squeeze(1), dim=1
        )

        diff = torch.mean(mean_top_k - mean_values)

        return diff


class ProtoWeightAlignment(nn.Module):
    def __init__(self):
        super(ProtoWeightAlignment, self).__init__()
        # we need 3 things for this to work
        # (1) sample ids (2) proto weight tensors (3) spikenet_channel_dict

        self.cossim = torch.nn.CosineSimilarity()

        self.name = "protoweight_alignment"
        self.required_forward_results = {
            "prototype_activations",
            "sample_ids",
            "spikenet_channel_dict",
            "proto_weight_tensors",
            "prototype_class_identity",
            "target",
        }

    def forward(
        self,
        prototype_activations: "prototype_activations",
        sample_ids: "sample_ids",
        spikenet_channel_dict: "spikenet_channel_dict",
        proto_weight_tensors: "proto_weight_tensors",
        prototype_class_identity: "prototype_class_identity",
        target: "target",
    ):
        """
        prototype_activations: [bsz, # protos, 1, 1] prototype activations
        sample_ids: list of eeg names in the batch
        spikenet_channel_dict: a dictionary of spikenet labels {eeg_name: [37]-channel labels}
        proto_weight_tensors: the prototype weight tensors

        """
        # grab the highest activated prototype for each input
        proto_acts = torch.argmax(
            prototype_activations[:, :].squeeze(-1).squeeze(-1), dim=-1
        )  # [bsz] output

        prototype_class_identity = torch.argmax(prototype_class_identity, dim=1)

        # for each input, the prototype with the highest activation's weights.
        # e.g. 1 row = input 8 highest act is proto 7, which has weights [37]..
        proto_weights_at_highest_act = proto_weight_tensors[
            proto_acts, 0
        ]  # [bsz,37] output
        proto_labels_at_highest_act = prototype_class_identity[proto_acts]

        # confirmed this creates the correct values given the sample_ids
        input_weights = []
        for i in sample_ids:
            # grab the actual sample spike probabilities
            input_weights.append(spikenet_channel_dict[i])

        # stack into torch
        input_weights = torch.stack(input_weights)

        device = proto_weights_at_highest_act.device
        sm_proto_weights = threshold_temperature_softmax(
            proto_weights_at_highest_act,
            proto_labels_at_highest_act,
            dim=1,
            temperature=0.9,
            threshold=0.015,
        )
        sm_input_weights = threshold_temperature_softmax(
            input_weights, target, dim=1, temperature=0.9, threshold=0.015
        )

        # MSE between input spike probs and highest activated proto spike probs
        cossim = self.cossim(sm_input_weights, sm_proto_weights.to(device))

        return -cossim.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.name = "contrastive"
        self.required_forward_results = {
            "latent_vectors",
            "proto_batch_dtw_tensor",
            "sample_ids",
        }
        self.mse = nn.MSELoss()

    def bucket_mapping(self, x):
        boundaries = torch.tensor([30, 40, 50, 70, 100, 125, 200])
        values = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.25])

        buckets = torch.bucketize(x, boundaries, right=True)
        return values[buckets]

    def forward(
        self,
        latent_vectors: "latent_vectors",
        proto_batch_dtw_tensor: "proto_batch_dtw_tensor",
        sample_ids: "sample_ids",
    ):
        # latent_vectors is [bsz, 258, 1, 37], we only want the latent which is first 128
        latent_chunk = latent_vectors[:, :128, 0, :]  # proto_batch_dtw_tensor

        # normalize along 128
        normalized_latent_chunk = torch.nn.functional.normalize(latent_chunk, dim=1)

        # og: [bsz, 128, 37] want bmm([37, BSZ, 128], [37, 128, BSZ])
        # want (1) is transpose(0,2).transpose(1,2) (2) is transpose (0,2)
        bmm1 = normalized_latent_chunk.transpose(0, 2).transpose(1, 2)
        bmm2 = normalized_latent_chunk.transpose(0, 2)

        arr = torch.bmm(bmm1, bmm2)  # shape [37, N, N] but we want [N,N,37]
        arr = arr.transpose(0, 2)  # (N,N,37) :)

        torch.save(proto_batch_dtw_tensor, "prebuckets_dtw.pth")

        bucketized_dtw = self.bucket_mapping(proto_batch_dtw_tensor)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bucketized_dtw = bucketized_dtw.to(device)
        arr = arr.to(device)

        mse_loss = self.mse(bucketized_dtw, arr)

        print("post bmm shape: ", arr.shape)
        print("dtw thing shape: ", batch_pairwise_dtw_tensor.shape)

        # create a mask with the DTW values
        assert batch_pairwise_dtw_tensor.shape == arr.shape

        N = arr.shape[0]

        # test 1 ensure that things diagonals are good
        for ii in range(N):
            for jj in range(N):
                if ii == jj:
                    assert arr[ii, jj].mean().item() == 1, print(
                        "failed diag: ", arr[ii, jj].mean()
                    )

        # test 2 ensure that things symmetry is good
        for ii in range(N):
            for jj in range(N):
                assert arr[ii, jj].mean() == arr[jj, ii].mean(), print(
                    "failed sym: ", arr[ii, jj].mean(), arr[jj, ii].mean()
                )

        # ensure that the DTW mask is correct
        torch.save(bucketized_dtw, "bucketized_dtw.pth")
        torch.save(sample_ids, "sample_ids.pth")

        # test 3 ensure that bmm is right
        if N > 15:
            import random

            rand_num = random.randint(0, 14)
            rand_num2 = random.randint(0, 14)

            # we want to ensure actual cossim is calculated
            # [bsz, 128, 37] shaped chunk
            a = normalized_latent_chunk[rand_num]
            b = normalized_latent_chunk[rand_num2]

            print("a,b shape: ", a.shape, b.shape)

            c = torch.sum(a * b, dim=0)
            print("c shape:", c.shape)

            d = arr[rand_num, rand_num2]
            print("d shape: ", d.shape)

            assert (
                torch.abs(c - d).sum() < 0.1
            ), "Tensors are not close enough (total difference >= 0.1)"

        print(n)

        print("constrastive loss: ", mse_loss)
        return mse_loss


class ImportanceStatsReg(nn.Module):
    def __init__(self):
        super(ImportanceStatsReg, self).__init__()
        self.name = "importance_stats"
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, model: "ProtoPNet"):
        # model.prototype_layer.importance_by_statistic [latent, range, var ,fft]

        sm_weights = self.softmax(model.prototype_layer.importance_by_statistic)

        latent_value = sm_weights[0]

        return latent_value - torch.min(sm_weights[1:])


class CrossEntropyCost(nn.Module):
    def __init__(self):
        super(CrossEntropyCost, self).__init__()
        self.name = "cross_entropy"
        self.required_forward_results = {"logits", "target"}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        target = target.float()
        return F.binary_cross_entropy_with_logits(logits[:, 0], target)


class WeightedIncorrectClassPrototypeActivations(nn.Module):
    def __init__(self):
        super(WeightedIncorrectClassPrototypeActivations, self).__init__()

    def forward(
        self,
        *,
        similarity_score_to_each_prototype,
        prototypes_of_wrong_class,
        target,
        prototype_class_identity,
        act_funct=None
    ):

        # target [bsz]
        # bsz x # protos x 37
        # proto classes: [# protos]
        # stack this [1 x # protos] - [target].unsqueeze(0)
        # class takes actiavtion function, do a sigmoid ontop (id for now)

        target = target.unsqueeze(1)  # shape [bsz, 1]
        prototype_ids = prototype_class_identity.argmax(dim=1).unsqueeze(
            0
        )  # shape [1,#protos]

        num_classes = prototype_class_identity.shape[1]
        class_scale = max(num_classes - 1, 1)
        weighted_arr = torch.abs(target - prototype_ids) / class_scale

        if act_funct != None:
            weighted_arr = act_funct(weighted_arr)

        incorrect_class_prototype_activations, _ = torch.max(
            similarity_score_to_each_prototype
            * prototypes_of_wrong_class
            * weighted_arr,
            dim=1,
        )

        return incorrect_class_prototype_activations


class IncorrectClassPrototypeActivations(nn.Module):
    def __init__(self):
        super(IncorrectClassPrototypeActivations, self).__init__()

    def forward(self, *, similarity_score_to_each_prototype, prototypes_of_wrong_class):
        incorrect_class_prototype_activations, _ = torch.max(
            similarity_score_to_each_prototype * prototypes_of_wrong_class, dim=1
        )

        return incorrect_class_prototype_activations


class CrossEntropyCost_OLDOLDOLD(nn.Module):
    def __init__(self):
        super(CrossEntropyCost, self).__init__()
        self.name = "cross_entropy"

        # TODO: Should these be functions or lists?
        self.required_forward_results = {"logits", "target"}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        cross_entropy = torch.nn.functional.cross_entropy(logits, target)
        return cross_entropy


class L1CostClassConnectionLayer(nn.Module):
    def __init__(self):
        super(L1CostClassConnectionLayer, self).__init__()
        self.name = "l1"

    def forward(self, model: "ProtoPNet"):
        l1 = model.prototype_prediction_head.class_connection_layer.weight.norm(p=1)
        return l1


class DTW_contrast(nn.Module):
    def __init__(self):
        super(DTW_contrast, self).__init__()
        self.name = "dtw_contrast"
        self.required_forward_results = {
            "nonweighted_simscores",
            "proto_dict",
            "all_dtw_tensor",
        }
        #  nonweighted_simscores torch.Size([90, 45, 1, 37])

    def forward(self, nonweighted_simscores, proto_dict, all_dtw_tensor):
        # nonweight simscore
        # torch.Size([bsz, # proto, 1, 37])

        # proto dict -- same as proto info dict used in local analysis
        # ex: {0;sample_id...}

        # dtw_dict [a][b]
        # {eeg1: {eeg_i: [37], eeg_j: [37], ...}, eeg_2 : {...}}

        len(proto_dict)
        nonweighted_simscores.shape[0]

        # if proto is empty, push hasn't happened so loss is 0
        if len(proto_dict) == 0:
            x = torch.tensor(0.0)
            return x

        high_mask = (all_dtw_tensor > 75).cuda()  # shape [bsz, # proto, 1, 37]

        if high_mask.sum() == 0:
            return torch.tensor(0.0)

        high_sum = (
            nonweighted_simscores * high_mask * all_dtw_tensor.cuda() / 100
        ).sum() / high_mask.sum()

        return torch.mean(torch.relu(high_sum))


class DTW_alike(nn.Module):
    def __init__(self):
        super(DTW_alike, self).__init__()
        self.name = "dtw_alike"
        self.required_forward_results = {
            "nonweighted_simscores",
            "proto_dict",
            "all_dtw_tensor",
        }
        #  nonweighted_simscores torch.Size([90, 45, 1, 37])

    def forward(self, nonweighted_simscores, proto_dict, all_dtw_tensor):
        # nonweight simscore
        # torch.Size([bsz, # proto, 1, 37])

        # proto dict -- same as proto info dict used in local analysis
        # ex: {0;sample_id...}

        # dtw_dict [a][b]
        # {eeg1: {eeg_i: [37], eeg_j: [37], ...}, eeg_2 : {...}}

        len(proto_dict)
        nonweighted_simscores.shape[0]

        # if proto is empty, push hasn't happened so loss is 0
        if len(proto_dict) == 0:
            x = torch.tensor(0.0)
            return x

        low_mask = (all_dtw_tensor < 45).cuda()

        if low_mask.sum() == 0:
            return torch.tensor(0.0)

        low_sum = -(nonweighted_simscores * low_mask).sum() / low_mask.sum()

        return torch.mean(low_sum)


class DTWLoss(nn.Module):
    def __init__(self):
        super(DTWLoss, self).__init__()
        self.name = "dtw"
        self.required_forward_results = {
            "nonweighted_simscores",
            "dtw_dict",
            "sample_ids",
            "proto_dict",
        }

        #   torch.Size([90, 45, 1, 37])

    def scale_dtw_scores(self, x):

        final = -0.05 * x + 2.5
        # x = (x/750 + 1) / (x/750 + 0.05)
        # final = torch.log(x) / torch.log(torch.tensor(10, dtype=x.dtype, device=x.device))

        return final.clamp_(min=0, max=1)

    def forward(self, nonweighted_simscores, dtw_dict, proto_dict, sample_ids):
        # nonweight simscore
        # torch.Size([bsz, # proto, 1, 37])

        # proto dict -- same as proto info dict used in local analysis
        # ex: {0;sample_id...}

        # dtw_dict [a][b]
        # {eeg1: {eeg_i: [37], eeg_j: [37], ...}, eeg_2 : {...}}

        num_protos = len(proto_dict)
        bsz_size = nonweighted_simscores.shape[0]

        # if proto is empty, push hasn't happened so loss is 0
        if len(proto_dict) == 0:
            x = torch.tensor(0.0)
            return x

        all_dtw_tensor = torch.zeros(bsz_size, num_protos, 1, 37)
        sample_index = 0

        for (
            sample_id
        ) in sample_ids:  # for 1 input sample, get all prototes that correspond to it
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
            all_dtw_tensor[sample_index, :, 0, :] = torch_dtw_scores
            sample_index += 1  # move onto next input

        ##### Save everything you need to test this #####
        """
        torch.save(proto_dict, "./CODETEST/prototype_dict_4test")
        torch.save(dtw_dict, "./CODETEST/dtw_dict_4test")
        torch.save(all_dtw_tensor, "./CODETEST/all_dtw_tensor_4test")
        torch.save(sample_ids, "./CODETEST/sample_ids_4test")
        """
        # all_dtw_tensor = self.scale_dtw_scores(all_dtw_tensor)
        # mse = torch.mean((nonweighted_simscores - all_dtw_tensor.cuda()) ** 2)

        high_mask = all_dtw_tensor > 130  # shape [bsz, # proto, 1, 37]

        low_mask = all_dtw_tensor < 50

        high_sum = (nonweighted_simscores * high_mask).sum()

        low_sum = -(nonweighted_simscores * low_mask).sum()

        return torch.mean(high_sum + low_sum)


class ClusterCost(nn.Module):
    def __init__(self, class_specific: bool = True):
        super(ClusterCost, self).__init__()
        self.class_specific = class_specific
        self.name = "cluster"

        self.required_forward_results = {
            "similarity_score_to_each_prototype",
            "prototypes_of_correct_class",
            "target",
        }

    def forward(
        self,
        target,
        similarity_score_to_each_prototype,
        prototypes_of_correct_class=None,
        act_fn=None,
    ):
        # Raise Assertion if similarity_score_to_each_prototype, prototypes_of_correct_class is 1D
        assert similarity_score_to_each_prototype.dim() > 1 and (
            prototypes_of_correct_class is None or prototypes_of_correct_class.dim() > 1
        ), "Max activations or prototypes of correct class is 1D."

        if self.class_specific:
            assert (
                prototypes_of_correct_class is not None
            ), "Prototypes of correct class must be provided to calculate cluster cost."

            # sim score = [90 x 45], protos_of_correct_class = [90 x 45], target = [90 x 1]
            num_classes = prototypes_of_correct_class.shape[1]
            class_scale = max(num_classes - 1, 1)
            weighted_target = target.unsqueeze(1) / class_scale

            if act_fn:
                weighted_target = act_fn(weighted_target)

            closest_sample_activations, _ = torch.max(
                weighted_target
                * similarity_score_to_each_prototype
                * prototypes_of_correct_class,
                dim=1,
            )

            # print("weighted: ", weighted_target.sum())
            # print("sim score: ", similarity_score_to_each_prototype.sum())
            # print("prototypes_of_correct_class: ", prototypes_of_correct_class.sum())
        else:
            closest_sample_activations, _ = torch.max(
                similarity_score_to_each_prototype, dim=1
            )

        cluster_cost = torch.mean(closest_sample_activations)
        assert self.class_specific == True
        # print("cluster cost: ", cluster_cost.item())

        return -cluster_cost


class WeightedSeparationCost(nn.Module):
    def __init__(self):
        super(WeightedSeparationCost, self).__init__()
        self.name = "separation"

        self.required_forward_results = {
            "incorrect_class_prototype_activations",
            "target",
            "prototype_class_identity",
        }

    def forward(
        self, incorrect_class_prototype_activations, target, prototype_class_identity
    ):
        if incorrect_class_prototype_activations is None:
            raise ValueError(
                "Incorrect class prototype activations must be provided to calculate separation cost"
            )

        prototype_ids = prototype_class_identity.argmax(dim=1)

        print("target: ", target, target.shape)
        print("proto ids: ", prototype_ids, prototype_ids.shape)

        print(
            "incorrect_class_prototype_activations: ",
            incorrect_class_prototype_activations.shape,
        )

        class_scale = max(prototype_class_identity.shape[1] - 1, 1)
        weighting = torch.abs(prototype_ids - target) / class_scale

        print("weighting: ", weighting.shape, weighting)
        print(
            "incorrect_class_prototype_activations: ",
            incorrect_class_prototype_activations.shape,
        )
        print(n)

        # incorrect_class_prototype_activations is shape [# protos], should be able just multiple [proto class id]
        separation_cost = torch.mean(incorrect_class_prototype_activations)

        return separation_cost


class SeparationCost(nn.Module):
    def __init__(self):
        super(SeparationCost, self).__init__()
        self.name = "separation"

        self.required_forward_results = {"incorrect_class_prototype_activations"}

    def forward(self, incorrect_class_prototype_activations):
        if incorrect_class_prototype_activations is None:
            raise ValueError(
                "Incorrect class prototype activations must be provided to calculate separation cost"
            )

        # incorrect_class_prototype_activations is shape [# protos], should be able just multiple [proto class id]
        separation_cost = torch.mean(incorrect_class_prototype_activations)

        # print("separatiohn cost: ", separation_cost.item())

        return separation_cost


class AverageSeparationCost(nn.Module):
    def __init__(self):
        super(AverageSeparationCost, self).__init__()
        self.name = "average_separation"

        self.required_forward_results = {
            "incorrect_class_prototype_activations",
            "prototypes_of_wrong_class",
        }

    def forward(
        self,
        incorrect_class_prototype_activations,
        prototypes_of_wrong_class=None,
    ):
        # Raise Assertion if prototypes_of_wrong_class is 1D
        assert prototypes_of_wrong_class.dim() > 1, "Prototypes of wrong class is 1D."

        if not (
            incorrect_class_prototype_activations is not None
            and prototypes_of_wrong_class is not None
        ):
            return None

        avg_separation_cost = incorrect_class_prototype_activations / torch.sum(
            prototypes_of_wrong_class, dim=1
        )

        avg_separation_cost = torch.mean(avg_separation_cost)

        return avg_separation_cost


class OffsetL2Cost(nn.Module):
    def __init__(self):
        super(OffsetL2Cost, self).__init__()

        self.name = "offset_l2"

    def forward(self, input_normalized: torch.Tensor, model: "ProtoPNet"):
        # Need to pass in input_normalized

        # TODO: Need conv_offset Sequential in skeleton for this to work
        # offsets = model.module.conv_offset(input_normalized)
        offsets = torch.ones_like(input_normalized)

        offset_l2 = offsets.norm()
        return offset_l2


class StackedOrthogonalityLoss(nn.Module):
    def __init__(self):
        super(StackedOrthogonalityLoss, self).__init__()
        self.name = "orthogonality_loss"

    def forward(self, model: "ProtoPNet"):
        ortho_loss = 0

        latent_inds = model.prototype_layer.activation_function.indices_for_latent
        fft_inds = model.prototype_layer.activation_function.indices_for_ffts

        prototype_tensors = model.prototype_layer.prototype_tensors

        for chunk_of_indices in [latent_inds, fft_inds]:

            selected_parts = prototype_tensors[:, chunk_of_indices].reshape(
                prototype_tensors.shape[0], -1
            )
            selected_parts = F.normalize(selected_parts, p=2, dim=-1)
            ortho = torch.mm(
                selected_parts, selected_parts.transpose(1, 0)
            ) - torch.eye(selected_parts.shape[0], device=selected_parts.device)

            ortho_loss = ortho_loss + torch.norm(ortho)
        return ortho_loss


class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()
        self.name = "orthogonality_loss"

    def forward(self, model: "ProtoPNet"):

        prototype_tensors = model.prototype_layer.prototype_tensors

        # delete this if not using protoEEGNet
        assert prototype_tensors.shape[1] == 128

        # Seperate prototypes out by class
        prototype_tensors = prototype_tensors.reshape(
            model.prototype_layer.num_prototypes_per_class,
            model.prototype_layer.num_classes,
            *prototype_tensors.shape[-3:],
        )

        # Permute and reshape these to (num_classes, protos_per_class*parts_per_proto, channel)
        prototype_tensors = prototype_tensors.permute(1, 0, 3, 4, 2).reshape(
            model.prototype_layer.num_classes, -1, prototype_tensors.shape[-3]
        )

        # Normalize each part to unit length
        prototype_tensors = F.normalize(prototype_tensors, p=2, dim=-1)

        # Get our (num_classes, protos_per_class*parts_per_proto, protos_per_class*parts_per_proto)
        # orthogonality matrix
        orthogonalities = torch.bmm(
            prototype_tensors, prototype_tensors.transpose(-2, -1)
        )

        # Subtract out the identity matrix
        orthogonalities = orthogonalities - torch.eye(
            orthogonalities.shape[-1], device=orthogonalities.device
        ).unsqueeze(0)

        # And compute our loss
        ortho_loss = torch.sum(torch.norm(orthogonalities, dim=(1, 2)))

        return ortho_loss


class SerialFineAnnotationCost(nn.Module):
    def __init__(self):
        super(SerialFineAnnotationCost, self).__init__()

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        upsampled_activation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
        white_coef=None,
    ):
        prototype_targets = prototype_class_identity.argmax(dim=1)
        v, i = prototype_targets.sort()
        if (v != prototype_targets).all():
            raise NotImplementedError(
                "Do not use Serial Fine Annotation cost when prototypes are not grouped together."
            )
        _, class_counts = prototype_targets.unique(return_counts=True)
        unique_counts = class_counts.unique()
        if len(unique_counts) != 1:
            raise NotImplementedError(
                "Do not use Serial Fine Annotation cost when prototype classes are imbalanced."
            )

        proto_num_per_class = list(set(class_counts))[0]
        device = upsampled_activation.device

        all_white_mask = torch.ones(
            upsampled_activation.shape[2], upsampled_activation.shape[3]
        ).to(device)

        fine_annotation_cost = 0

        for index in range(target.shape[0]):
            weight1 = 1 * all_white_mask
            weight2 = 1 * fine_annotation[index]

            if white_coef is not None:
                weight1 *= white_coef

            fine_annotation_cost += (
                torch.norm(
                    upsampled_activation[index, : target[index] * proto_num_per_class]
                    * (weight1)
                )
                + torch.norm(
                    upsampled_activation[
                        index,
                        target[index]
                        * proto_num_per_class : (target[index] + 1)
                        * proto_num_per_class,
                    ]
                    * (weight2)
                )
                + torch.norm(
                    upsampled_activation[
                        index,
                        (target[index] + 1) * proto_num_per_class :,
                    ]
                    * (weight1)
                )
            )

        return fine_annotation_cost


class GenericFineAnnotationCost(nn.Module):
    def __init__(self, scoring_function):
        """
        Parameters:
        ----------
        scoring_function (function): Function for aggregating the loss costs. Will receive the masked activations.
        """
        super(GenericFineAnnotationCost, self).__init__()
        self.scoring_function = scoring_function

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        upsampled_activation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
    ):
        """
        Calculates the fine-annotation loss for a given set of inputs.

        Parameters:
        ----------
            target (torch.Tensor): Tensor of targets. Size(Batch)
            upsampled_activation (torch.Tensor): Size(batch, n_prototypes, height, width)
            fine_annotation (torch.Tensor): Fine annotation tensor Size(batch, 1, height, width)
            prototype_class_identity (torch.Tensor): Class identity tensor for prototypes size(num_prototypes, num_classes)

        Returns:
        --------
            fine_annotation_loss (torch.Tensor): Fine annotation loss tensor

         Notes:
        -----
            This function assumes that the input tensors are properly aligned such that the prototype at index i
            in the `upsampled_activation` tensor corresponds to the class at index i in the `prototype_class_identity`
            tensor.

        Called in following files:
            - train_and_eval.py: l2_fine_annotation_loss(), square_fine_annotation_loss()

        """
        target_set = target.unique()
        class_fa_losses = torch.zeros(target_set.shape[0])

        # Assigned but never used in IAIA-BL
        # total_proto = upsampled_activation.shape[1]

        # unhot the one-hot encoding
        prototype_targets = prototype_class_identity.argmax(
            dim=1
        )  # shape: (n_prototype)

        # This shifts our iteration from O(n) to O(#targets)
        for target_val in list(target_set):
            # We have different calculations depending on whether or not the prototype
            # is in class or not, so we will find each group
            in_class_targets = target == target_val  # shape: (batch)
            in_class_prototypes = (
                prototype_targets == target_val
            )  # shape: (n_prototypes)

            # In Class case Size(D', p=y, 244, 244)
            prototype_activation_in_class = upsampled_activation[in_class_targets][
                :, in_class_prototypes, :, :
            ]
            # broadcast fine_annotation to prototypes in dim 1
            prototypes_activation_in_class_masked = (
                prototype_activation_in_class * fine_annotation[in_class_targets]
            )

            # Out of class case Size(batch, p!=y, 244, 244)
            prototype_activation_out_of_class = upsampled_activation[in_class_targets][
                :, ~in_class_prototypes, :, :
            ]

            # regroup after masking to parallelize, Size(batch, p, 244, 244)
            class_activations = torch.cat(
                (
                    prototypes_activation_in_class_masked,
                    prototype_activation_out_of_class,
                ),
                1,
            )

            # Size(D', p) - norms for all prototypes
            class_fa_for_all_prototypes = self.scoring_function(class_activations)

            class_fa_losses[target_val] = torch.sum(class_fa_for_all_prototypes)

        fine_annotation_loss = class_fa_losses.sum()
        return fine_annotation_loss


class FineAnnotationCost(nn.Module):
    def __init__(self, fa_loss: str = "serial"):
        super(FineAnnotationCost, self).__init__()

        self.fa_loss = fa_loss
        self.name = "fine_annotation"
        self.required_forward_results = {
            "target",
            "fine_annotation",
            "upsampled_activation",
            "prototype_class_identity",
        }

        # TODO: Could choose just one cost function here
        # And then determine necessary parameters as kwdict
        # Make it more generic
        self.serial_cost = SerialFineAnnotationCost()
        self.l2_fine_annotation_loss = GenericFineAnnotationCost(self.l2_scoring)
        self.square_fine_annotation_loss = GenericFineAnnotationCost(
            self.square_scoring
        )

        assert self.fa_loss in ["serial", "l2_norm", "square"]

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        upsampled_activation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
    ):
        target = torch.tensor(target).int()
        if fine_annotation is None:
            fa_shape = upsampled_activation.shape
            fa_shape[1] = 1
            fine_annotation = torch.zero(fa_shape)
        if self.fa_loss == "serial":
            fine_annotation_cost = self.serial_cost(
                target, fine_annotation, upsampled_activation, prototype_class_identity
            )
        elif self.fa_loss == "l2_norm":
            fine_annotation_cost = self.l2_fine_annotation_loss(
                target,
                fine_annotation,
                upsampled_activation,
                prototype_class_identity,
            )
        elif self.fa_loss == "square":
            fine_annotation_cost = self.square_fine_annotation_loss(
                target,
                fine_annotation,
                upsampled_activation,
                prototype_class_identity,
            )

        return fine_annotation_cost

    def l2_scoring(self, activations):
        return activations.norm(p=2, dim=(2, 3))

    def square_scoring(self, activations):
        return activations.square().sum(dim=(2, 3))
