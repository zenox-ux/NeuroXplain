import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

print("[TRACE] activations.py module loaded.")

class CosPrototypeActivation:
    """
    Computes the cosine activation (arc distance) between convolutional features as in
        https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(
        self,
        relu_on_cos: bool = True,
        normalization_constant: int = 64,
        episilon_val: float = 1e-4,
    ):
        """
        Args:
            margin: Margin for the cosine similarity. If None, then no margin is used.
            relu_on_cos: Whether to apply a ReLU on the cosine similarity. If False, then the cosine similarity is
                returned as is.
            normalization_constant: The normalization constant for the cosine similarity. This is used to scale the
                cosine similarity to a reasonable range. The default value of 64 is chosen to be consistent with the
                original ProtoPNet implementation.
            episilon_val: A small value to prevent division by zero.
        """
        self.relu_on_cos = relu_on_cos
        self.epsilon_val = episilon_val
        self.input_vector_length = normalization_constant
        self.normalization_constant = normalization_constant
        print(f"[TRACE] CosPrototypeActivation initialized: relu={relu_on_cos}, norm_const={normalization_constant}")

    def _normalize(self, x: torch.Tensor, prototype_tensor: torch.Tensor):
        normalizing_factor = (
            prototype_tensor.shape[-2] * prototype_tensor.shape[-1]
        ) ** 0.5

        x_length = torch.sqrt(torch.sum(torch.square(x), dim=-3) + self.epsilon_val)
        x_length = x_length.view(
            x_length.size()[0], 1, x_length.size()[1], x_length.size()[2]
        )
        x_normalized = self.normalization_constant * x / x_length
        x_normalized = x_normalized / normalizing_factor
        return x_normalized, normalizing_factor

    def __call__(
        self,
        x: torch.Tensor,
        prototype_tensor: torch.Tensor,
    ):
        """
        Args:
            x: The input tensor of shape (batch_size, feature_dim, latent_height, latent_width)
            prototype_tensor: The prototype tensor of shape (num_prototypes, feature_dim, latent_height, latent_width)
            prototypes_of_wrong_class: The prototypes of the wrong class. This is used for the margin loss.

        Returns: activations (torch.Tensor): Tensor of the activations. This is of shape (batch_size, num_prototypes, activation_height, activation_width).
        """
        print(f"[TRACE] CosPrototypeActivation __call__ | x shape: {list(x.shape)} | proto shape: {list(prototype_tensor.shape)}")
        
        x_normalized, normalizing_factor = self._normalize(x, prototype_tensor)

        # We normalize prototypes to unit length
        prototype_vector_length = torch.sqrt(
            torch.sum(torch.square(prototype_tensor), dim=-3) + self.epsilon_val
        )
        prototype_vector_length = prototype_vector_length.view(
            prototype_vector_length.size()[0],
            1,
            prototype_vector_length.size()[1],
            prototype_vector_length.size()[2],
        )
        normalized_prototypes = prototype_tensor / (
            prototype_vector_length + self.epsilon_val
        )
        normalized_prototypes = normalized_prototypes / normalizing_factor

        if x_normalized.device != normalized_prototypes.device:
            normalized_prototypes = normalized_prototypes.to(x_normalized.device)
            
        activations_dot = F.conv2d(x_normalized, normalized_prototypes)
        print(f"[TRACE]   > Raw Dot Product mean: {activations_dot.mean().item():.4f}")

        renormed_activations = activations_dot / (self.normalization_constant * 1.01)

        if self.relu_on_cos:
            renormed_activations = torch.relu(renormed_activations)

        print(f"[TRACE]   > Final Activation mean: {renormed_activations.mean().item():.4f}")
        return renormed_activations


class CosineSimilarityWithStats:

    def __init__(
        self,
        indices_for_latent=[i for i in range(128)],
        indices_for_ranges=[128],
        indices_for_vars=[129],
        indices_for_ffts=[130 + i for i in range(128)],
        range_buckets=None,
        var_buckets=None,
    ):

        self.indices_for_latent = indices_for_latent
        self.indices_for_ranges = indices_for_ranges
        self.indices_for_vars = indices_for_vars
        self.indices_for_ffts = indices_for_ffts

        self.range_buckets = range_buckets
        self.var_buckets = var_buckets
        print(f"[TRACE] CosineSimilarityWithStats initialized with indices mapping.")

    def __convert_to_percentile(self, x=None, boundaries=None, name=None):
        """
        args:
        x: input to bucketize
        boudaries: bounds in which to place x

        """
        device = x.device
        outputs = (torch.bucketize(x, boundaries.to(device)) - 1) * 0.01
        return outputs

    def __minmax_normalize(self, x, min, max):
        device = x.device
        # Use existing tensors if possible to avoid UserWarnings about copy construction
        t_min = min if torch.is_tensor(min) else torch.tensor(min)
        t_max = max if torch.is_tensor(max) else torch.tensor(max)
        
        t_min = t_min.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        t_max = t_max.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        return (x - t_min) / (t_max - t_min + 0.001)

    def __dist_to_sim(self, x, min=200):
        result = 800 / (x + 0.001)
        return torch.clamp(result, max=1)

    def get_summary_stats(self, x, prototype_tensors):
        print(f"[TRACE]   > Calculating Summary Stats (Range, Var, FFT)...")
        range_sims = torch.abs(
            (
                self.__minmax_normalize(
                    x[:, self.indices_for_ranges].unsqueeze(1),
                    min=self.range_buckets[3],
                    max=self.range_buckets[98],
                )
                - self.__minmax_normalize(
                    prototype_tensors[:, self.indices_for_ranges].unsqueeze(0),
                    min=self.range_buckets[3],
                    max=self.range_buckets[98],
                )
            )
        )

        range_sims = 1 - range_sims.squeeze(-2)
        range_sims = torch.clamp(range_sims, max=1, min=0)
        print(f"[TRACE]     >> Range Sims mean: {range_sims.mean().item():.4f}")

        # range_sims is (batch, num_protos, 1, num_input_channels)
        var_sims = torch.abs(
            (
                self.__minmax_normalize(
                    x[:, self.indices_for_vars].unsqueeze(1),
                    min=self.var_buckets[3],
                    max=self.var_buckets[97],
                )
                - self.__minmax_normalize(
                    prototype_tensors[:, self.indices_for_vars].unsqueeze(0),
                    min=self.var_buckets[3],
                    max=self.var_buckets[97],
                )
            )
        )
        var_sims = 1 - var_sims.squeeze(-2)
        var_sims = torch.clamp(var_sims, max=1, min=0.0)
        print(f"[TRACE]     >> Var Sims mean: {var_sims.mean().item():.4f}")

        fft_sims = torch.norm(
            x[:, self.indices_for_ffts].unsqueeze(1)
            - prototype_tensors[:, self.indices_for_ffts].unsqueeze(0),
            dim=-3,
        )

        fft_sims = self.__dist_to_sim(fft_sims)
        print(f"[TRACE]     >> FFT Sims mean: {fft_sims.mean().item():.4f}")

        outputs = {
            "range_sims": range_sims.float(),
            "var_sims": var_sims.float(),
            "fft_sims": fft_sims.float(),
        }

        return outputs

    def __call__(
        self,
        x: torch.Tensor,
        prototype_tensors: torch.Tensor,
        importance_by_statistic: torch.Tensor = torch.tensor(
            [0.35, 0.1, 0.1, 0.1, 0.1]
        ),
    ):
        """
        x -- torch.Tensor, shape (batch, latent_dim + num_stats, 1,  37)
        prototype_tensors -- torch.Tensor, shape (num_prototypes, latent_dim + num_stats, 1,  37)
        """
        print(f"[TRACE] CosineSimilarityWithStats __call__ triggered")
        assert torch.abs(torch.sum(importance_by_statistic) - 1.0) < 1e-5


class WeightedCosineSimilarityWithStats(CosineSimilarityWithStats):

    def __init__(
        self,
        indices_for_latent=[i for i in range(128)],
        indices_for_ranges=[128],
        indices_for_vars=[129],
        indices_for_ffts=[130 + i for i in range(128)],
        range_buckets=None,
        var_buckets=None,
    ):
        import pickle
        
        stats_path = "model_feats/quantile_summary_stats.pkl"
        print(f"[TRACE] WeightedCosineSimilarityWithStats loading stats from {stats_path}")
        with open(stats_path, "rb") as file:
            bounds_dict = pickle.load(file)

        super(WeightedCosineSimilarityWithStats, self).__init__(
            indices_for_latent=[i for i in range(128)],
            indices_for_ranges=[128],
            indices_for_vars=[129],
            indices_for_ffts=[130 + i for i in range(128)],
            range_buckets=bounds_dict["full"]["range"],
            var_buckets=bounds_dict["full"]["var"],
        )

    def __call__(
        self,
        x: torch.Tensor,
        prototype_tensors: torch.Tensor,
        weight_tensors: torch.Tensor,
        proto_class_identity: torch.Tensor,
        importance_by_statistic: torch.Tensor = torch.tensor([0.65, 0.1, 0.1, 0.05]),
    ):
        print(f"\n[TRACE] --- WeightedCosineSimilarity Calculation ---")
        device = x.device

        # check the dim is along the 128 dimension
        normalized_prototypes = torch.nn.functional.normalize(
            prototype_tensors[:, self.indices_for_latent], dim=1
        )
        normalized_inputs = torch.nn.functional.normalize(
            x[:, self.indices_for_latent], dim=1
        )

        proto_class_identity_idx = torch.argmax(proto_class_identity, dim=1)

        # Calculate Latent Shape Similarity
        prototype_dot_prod = torch.einsum(
            "bdzc,pdzc->bpzc",
            normalized_inputs[:, self.indices_for_latent],
            normalized_prototypes[:, self.indices_for_latent],
        )
        print(f"[TRACE]   > Latent Dot Prod mean: {prototype_dot_prod.mean().item():.4f}, std: {prototype_dot_prod.std().item():.4f}")

        # Get Stats Sims (Range, Var, FFT)
        outputs = super().get_summary_stats(x, prototype_tensors)

        a = importance_by_statistic[0]
        b = importance_by_statistic[1]
        c = importance_by_statistic[2]
        d = importance_by_statistic[3]
        print(f"[TRACE]   > Importance Weights: Latent={a:.4f}, Range={b:.4f}, Var={c:.4f}, FFT={d:.4f}")

        # Sum the similarities
        summed_activation = (
            a * prototype_dot_prod
            + b * outputs["range_sims"]
            + c * outputs["var_sims"]
            + d * outputs["fft_sims"]
        )
        print(f"[TRACE]   > Summed similarity mean (pre-channel-weights): {summed_activation.mean().item():.4f}")

        # Apply channel-wise weights (The mask)
        # weight_tensors is [bsz, 1, 37]
        prototype_activations = torch.einsum(
            "bpzc,bzc->bp", summed_activation, weight_tensors.to(device)
        )
        
        print(f"[TRACE]   > Final activations mean: {prototype_activations.mean().item():.4f}, std: {prototype_activations.std().item():.4f}")
        if prototype_activations.std() < 1e-5:
            print("[TRACE]   !!! WARNING: Standard Deviation is nearly zero. Possible Mode Collapse detected.")

        return {
            "prototype_activations": prototype_activations.unsqueeze(-1).unsqueeze(-1),
            "nonweighted_simscores": prototype_dot_prod,
            "summed_activation": summed_activation,
            "range_sims": outputs["range_sims"],
            "var_sims": outputs["var_sims"],
            "fft_sims": outputs["fft_sims"],
        }


class L2Activation:
    def __init__(self, epsilon_val=1e-4):
        self.epsilon_val = epsilon_val
        print(f"[TRACE] L2Activation initialized: epsilon={epsilon_val}")

    def __call__(
        self,
        x: torch.Tensor,
        prototype_tensors: torch.Tensor,
    ):
        print(f"[TRACE] L2Activation __call__ | x shape: {list(x.shape)}")
        x2 = x**2
        ones = torch.ones(prototype_tensors.shape, requires_grad=False).to(
            prototype_tensors.device
        )

        x2_patch_sum = F.conv2d(input=x2, weight=ones)

        p2 = prototype_tensors**2
        # TODO: Support more dimensions
        p2 = torch.sum(p2, dim=(1, 2, 3))

        # Reshape from (num_prototypes,) to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_tensors)
        intermediate_result = -2 * xp + p2_reshape

        distances = x2_patch_sum + intermediate_result

        distances = F.relu(distances)
        activations = torch.log((distances + 1) / (distances + self.epsilon_val))
        print(f"[TRACE]   > L2 Distances mean: {distances.mean().item():.4f}, Max act: {activations.max().item():.4f}")

        return activations


class ConvolutionalSharedOffsetPred(nn.Module):
    """
    Computes the activation for a deformable prototype as in
        https://arxiv.org/pdf/1801.07698.pdf, but with renormalization
        after deformation instead of norm-preserving interpolation.
    """

    def __init__(
        self,
        prototype_shape: tuple,
        input_feature_dim: int = 512,
        kernel_size: int = 3,
        prototype_dilation: int = 1,
    ):
        """
        Args:
            prototype_shape: The shape of the prototypes the convolution will be applied to
            input_feature_dim: The expected latent dimension of the input
            kernel_size: The size of the kernel used for offset prediction
        """
        assert (kernel_size % 2 == 1) or (
            prototype_dilation % 2 == 0
        ), f"Error: kernel size {kernel_size} with dilation {prototype_dilation} is not supported because even kernel sizes without even dilation break symmetric padding"
        assert (
            len(prototype_shape) == 4
        ), "Error: Code assumes prototype_shape is a (num_protos, channel, height, width) tuple."

        super(ConvolutionalSharedOffsetPred, self).__init__()
        self.prototype_shape = prototype_shape

        self.prototype_dilation = prototype_dilation

        # Compute out channels as 2 * proto_h * proto_w
        out_channels = 2 * prototype_shape[2] * prototype_shape[3]
        self.offset_predictor = torch.nn.Conv2d(
            input_feature_dim,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        torch.nn.init.zeros_(self.offset_predictor.weight)
        self._init_offset_bias()
        print(f"[TRACE] ConvolutionalSharedOffsetPred initialized: out_channels={out_channels}")

    def _init_offset_bias(self):
        # out channels is ordered as (tl_x, tl_y, tm_x, tm_y, ...)
        # Initialize our offset predictor to put us at normal grid sample locations
        new_bias = torch.zeros_like(self.offset_predictor.bias)
        for py in range(self.prototype_shape[-2]):
            for px in range(self.prototype_shape[-1]):
                new_bias[(py * self.prototype_shape[-2] + px) * 2 + 1] = (
                    self.prototype_dilation * (py - (self.prototype_shape[-2] - 1) / 2)
                )
                new_bias[(py * self.prototype_shape[-2] + px) * 2] = (
                    self.prototype_dilation * (px - (self.prototype_shape[-1] - 1) / 2)
                )

        self.offset_predictor.bias = torch.nn.Parameter(new_bias).to(
            self.offset_predictor.bias.device
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Args:
            x: The input tensor of shape (batch_size, feature_dim, latent_height, latent_width)

        Returns: activations (torch.Tensor): Tensor of the activations. This is of shape (batch_size, num_prototypes, activation_height, activation_width).
        """
        # predicted_offsets will be (batch, 2 * proto_h * proto_w, height, width)
        predicted_offsets = self.offset_predictor(x)
        return predicted_offsets