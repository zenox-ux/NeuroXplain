import torch
import numpy as np
from protopnet.eval_utils import knn_replace_step


class ProtoEEGkNN:
    def __init__(self, model_path, topk=10):
        """
        Initialize the KNN model inference class.

        Args:
            model_path (str): Path to the saved model
            knn_replace_step_func (callable): Function to convert model to KNN version
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # Load on CPU safely, then move (works for CPU-only torch too)
        self.base_model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.base_model.cpu()

        self.model, self.proto_labels, self.knn_proto_ids = knn_replace_step(
            self.base_model, model_path
        )

        self.model.eval().to(self.device)
        self.topk = topk

    def forward(self, eegs, input_ids):
        """
        Perform forward pass on test data and return predictions and labels.

        Args:
            test_loader: DataLoader containing test samples

        Returns:
            tuple: (y_true, y_pred) - normalized true labels and predictions
        """
        predictions = []
        matches = []

        # Forward pass through model
        latent_vectors = self.model.backbone(eegs.to(self.device))
        latent_vectors = self.model.add_on_layers(latent_vectors)
        output_dict = self.model.prototype_layer(latent_vectors, sample_ids=input_ids)
        proto_acts = output_dict["prototype_activations"].cpu().clone()

        if eegs.abs().max() < 1e-5:
            print(f"[ALERT] Batch {input_ids[0]} is ALL ZEROS before entering model!")
        # Get top 10 prototype activations
        *_, topk = torch.topk(proto_acts.squeeze(-1).squeeze(-1), k=self.topk, dim=1)

        # Calculate predictions based on closest prototype matches
        for i in range(topk.shape[0]):
            closest_matches = []
            proto_filenames = []
            for j in topk[i]:
                closest_matches.append(self.proto_labels[j])
                proto_filenames.append(self.knn_proto_ids[j])
            prediction = sum(closest_matches) / len(closest_matches)
            predictions.append(prediction)
            matches.append(proto_filenames)

        y_pred = np.array(predictions)
        output_dict["prediction"] = y_pred
        output_dict["matches"] = matches

        return output_dict
