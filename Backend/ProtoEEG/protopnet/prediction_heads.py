import torch
from torch import nn


class PrototypePredictionHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        prototype_class_identity: torch.Tensor,
        incorrect_class_connection: float = -0.5,
        k_for_topk: int = 1,
    ):
        super(PrototypePredictionHead, self).__init__()

        self.num_classes = num_classes
        self.incorrect_class_connection = incorrect_class_connection
        self.k_for_topk = k_for_topk
        self.prototype_class_identity = prototype_class_identity

        self.num_prototypes = prototype_class_identity.shape[0]
        self.class_connection_layer = nn.Linear(
            self.num_prototypes,
            self.num_classes,
            bias=False,
        )

        self.__set_last_layer_incorrect_connection()

    def __set_last_layer_incorrect_connection(self):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = self.incorrect_class_connection
        self.class_connection_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def forward(
        self,
        prototype_activations: torch.Tensor,
        upsampled_activation: torch.Tensor,
        **kwargs,
    ):
        # TODO: Update prototype_activations to be

        _activations = prototype_activations.view(
            prototype_activations.shape[0], prototype_activations.shape[1], -1
        )

        # When k=1, this reduces to the maximum
        k_for_topk = min(self.k_for_topk, _activations.shape[-1])
        topk_activations, _ = torch.topk(_activations, k_for_topk, dim=-1)
        similarity_score_to_each_prototype = torch.mean(topk_activations, dim=-1)

        logits = self.class_connection_layer(similarity_score_to_each_prototype)

        # output_dict = {
        #     "logits": logits,
        #     "similarity_score_to_each_prototype": similarity_score_to_each_prototype,
        #     "upsampled_activation": upsampled_activation,
        # }

        output_dict = {"logits": logits}

        if (
            "return_similarity_score_to_each_prototype" in kwargs
            and kwargs["return_similarity_score_to_each_prototype"]
        ) or (
            "return_incorrect_class_prototype_activations" in kwargs
            and kwargs["return_incorrect_class_prototype_activations"]
        ):
            output_dict["similarity_score_to_each_prototype"] = (
                similarity_score_to_each_prototype
            )

        if (
            "return_upsampled_activation" in kwargs
            and kwargs["return_upsampled_activation"]
        ):
            output_dict["upsampled_activation"] = upsampled_activation

        return output_dict


class PrototypeBinaryClassificationPredictionHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        prototype_class_identity: torch.Tensor,
        incorrect_class_connection: float = -0.5,
        k_for_topk: int = 1,
        bias=-1,
    ):
        super(PrototypeBinaryClassificationPredictionHead, self).__init__()

        self.num_classes = num_classes
        self.incorrect_class_connection = incorrect_class_connection
        self.k_for_topk = k_for_topk
        self.prototype_class_identity = prototype_class_identity
        self.bias = bias

        self.num_prototypes = prototype_class_identity.shape[0]
        self.class_connection_layer = nn.Linear(
            self.num_prototypes,
            1,
            bias=True,
        )

        self.__set_last_layer_incorrect_connection_eeg()

        # Set requires_grad to False for weights
        self.class_connection_layer.weight.requires_grad = False

        # Set requires_grad to True for bias (although it's True by default)
        self.class_connection_layer.bias.requires_grad = True

    def __set_last_layer_incorrect_connection_eeg(self):
        """
        Sets the last layer weights to be 0-1
        and the bias of the last layer to equal
        to the value specified in the config file


        """
        if self.num_classes != 2:
            raise ValueError(
                "PrototypeBinaryClassificationPredictionHead expects exactly 2 classes."
            )

        proto_class_ids = torch.argmax(self.prototype_class_identity, dim=1)
        negative_class = (proto_class_ids == 0).float()
        positive_class = (proto_class_ids == 1).float()
        my_weight_tensor = (positive_class - negative_class).unsqueeze(0)
        my_bias = torch.tensor([self.bias], dtype=my_weight_tensor.dtype).detach()

        self.class_connection_layer.weight.data.copy_(my_weight_tensor)
        self.class_connection_layer.bias.data.copy_(my_bias)

    def __set_last_layer_incorrect_connection_eeg_linear(self):
        """
        Legacy linear initializer kept for compatibility; now binary-only.
        """
        if self.num_classes != 2:
            raise ValueError(
                "PrototypeBinaryClassificationPredictionHead expects exactly 2 classes."
            )
        labels = [0.0, 1.0]

        proto_per_class = []
        for label in labels:
            for i in range(int(self.num_prototypes / self.num_classes)):
                proto_per_class.append(label)

        my_weight_tensor = torch.tensor(proto_per_class).unsqueeze(0)
        my_bias = torch.tensor([self.bias]).detach()

        self.class_connection_layer.weight.data.copy_(my_weight_tensor)
        self.class_connection_layer.bias.data.copy_(my_bias)

    def forward(
        self,
        prototype_activations: torch.Tensor,
        upsampled_activation: torch.Tensor,
        **kwargs,
    ):
        # TODO: Update prototype_activations to be
        # print("proto acts: ", prototype_activations.sum())
        _activations = prototype_activations.view(
            prototype_activations.shape[0], prototype_activations.shape[1], -1
        )

        # When k=1, this reduces to the maximum
        k_for_topk = min(self.k_for_topk, _activations.shape[-1])
        # print("k_for_top_k: ", k_for_topk)
        # print("_activations: ", _activations.shape, _activations.sum())
        topk_activations, _ = torch.topk(_activations, k_for_topk, dim=-1)
        similarity_score_to_each_prototype = torch.mean(topk_activations, dim=-1)

        logits = self.class_connection_layer(similarity_score_to_each_prototype)

        # print("cc layer weight: ", self.class_connection_layer.weight)
        # print("cc layer bias: ", self.class_connection_layer.bias)

        # output_dict = {
        #     "logits": logits,
        #     "similarity_score_to_each_prototype": similarity_score_to_each_prototype,
        #     "upsampled_activation": upsampled_activation,
        # }

        output_dict = {"logits": logits}

        if (
            "return_similarity_score_to_each_prototype" in kwargs
            and kwargs["return_similarity_score_to_each_prototype"]
        ) or (
            "return_incorrect_class_prototype_activations" in kwargs
            and kwargs["return_incorrect_class_prototype_activations"]
        ):
            output_dict["similarity_score_to_each_prototype"] = (
                similarity_score_to_each_prototype
            )

        if (
            "return_upsampled_activation" in kwargs
            and kwargs["return_upsampled_activation"]
        ):
            output_dict["upsampled_activation"] = upsampled_activation

        return output_dict
