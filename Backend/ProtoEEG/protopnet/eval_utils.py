from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    r2_score,
    precision_recall_curve,
    auc,
)
from protopnet.spikenet_helpers import EEG_ConcatDataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.utils import resample
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.utils import resample

print("[EVAL_UTILS] Imports completed.")

def bootstrap_metrics_ci(y_true, y_pred, n_iterations=10000):
    """
    Calculate original R², accuracy,and AUROC with 95% confidence intervals via bootstrapping.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    n_iterations : int, default=10000
        Number of bootstrap iterations

    Returns:
    --------
    dict : Dictionary containing original metrics and their 95% confidence intervals
    """
    print(f"\n--- ENTERING bootstrap_metrics_ci ---")
    # Ensure numpy arrays
    y_true = np.array(y_true)
    print(f"[LINE 26] y_true = np.array(y_true) | Type: {type(y_true)}, Shape: {y_true.shape}")
    y_pred = np.array(y_pred)
    print(f"[LINE 27] y_pred = np.array(y_pred) | Type: {type(y_pred)}, Shape: {y_pred.shape}")
    
    print(f"[DEBUG] y_true sample (first 5): {y_true[:5]}")
    print(f"[DEBUG] y_pred sample (first 5): {y_pred[:5]}")

    # Calculate original metrics
    r2_score(y_true, y_pred)
    print(f"[LINE 33] r2_score(y_true, y_pred) calculated (result discarded)")

    # Binary classification metrics
    y_true_binary = (y_true >= 0.5).astype(int)
    print(f"[LINE 35] y_true_binary = (y_true >= 0.5).astype(int) | Sum positive: {np.sum(y_true_binary)}")
    y_pred_binary = (y_pred >= 0.15).astype(int)
    print(f"[LINE 36] y_pred_binary = (y_pred >= 0.5).astype(int) | Sum positive: {np.sum(y_pred_binary)}")

    # Calculate original accuracy
    accuracy_score(y_true_binary, y_pred_binary)
    print(f"[LINE 39] accuracy_score(y_true_binary, y_pred_binary) calculated (result discarded)")

    # Calculate original AUROC
    original_auroc = roc_auc_score(y_true_binary, y_pred)
    print(f"[LINE 42] original_auroc = roc_auc_score(y_true_binary, y_pred) | Value: {original_auroc:.4f}")

    # Initialize lists to store bootstrap results
    bootstrap_r2 = []
    bootstrap_accuracy = []
    bootstrap_auroc = []
    print(f"[LINE 45-47] Iteration lists initialized.")

    # Bootstrap iterations
    print(f"[PROGRESS] Starting {n_iterations} bootstrap iterations...")
    for i in range(n_iterations):
        # Generate bootstrap sample indices
        indices = resample(range(len(y_true)), replace=True, n_samples=len(y_true))

        # Get bootstrap samples
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Calculate R² on bootstrap sample
        boot_r2 = r2_score(y_true_boot, y_pred_boot)
        bootstrap_r2.append(boot_r2)

        # Calculate binary metrics on bootstrap sample
        y_true_binary_boot = (y_true_boot >= 0.5).astype(int)
        y_pred_binary_boot = (y_pred_boot >= 0.5).astype(int)

        # Calculate accuracy on bootstrap sample
        boot_accuracy = accuracy_score(y_true_binary_boot, y_pred_binary_boot)
        bootstrap_accuracy.append(boot_accuracy)

        # Calculate AUROC on bootstrap sample
        boot_auroc = roc_auc_score(y_true_binary_boot, y_pred_boot)
        bootstrap_auroc.append(boot_auroc)
    
    print(f"[LINE 70] Bootstrap iterations finished.")

    # Calculate 95% confidence intervals
    np.percentile(bootstrap_r2, [2.5, 97.5])
    print(f"[LINE 73] np.percentile(bootstrap_r2, [2.5, 97.5]) calculated (result discarded)")
    np.percentile(bootstrap_accuracy, [2.5, 97.5])
    print(f"[LINE 74] np.percentile(bootstrap_accuracy, [2.5, 97.5]) calculated (result discarded)")
    auroc_ci = np.percentile(bootstrap_auroc, [2.5, 97.5])
    print(f"[LINE 75] auroc_ci = np.percentile(bootstrap_auroc, [2.5, 97.5]) | Value: {auroc_ci}")

    # Return results
    return {
    "r2": r2_score(y_true, y_pred),
    "r2_ci": np.percentile(bootstrap_r2, [2.5, 97.5]),
    "accuracy": accuracy_score(y_true_binary, y_pred_binary),
    "accuracy_ci": np.percentile(bootstrap_accuracy, [2.5, 97.5]),
    "auroc": original_auroc,
    "auroc_ci": auroc_ci,
    }


def knn_replace_step(model, model_path):
    print(f"\n--- ENTERING knn_replace_step ---")
    knn_data_name = "_binary"
    recalc_knn = True

    if model_path != None:
        name = model_path.split("/")[-2] + "_" + model_path.split("/")[-1][:-4]
        cache_file = f"./model_knn_layers/{name}{knn_data_name}.pth"
        print(f"[LINE 92-94] Generated cache path: {cache_file}")

        if os.path.exists(cache_file):
            print(f"[LINE 96] Cache exists. Loading: {cache_file}")
            model_dict = torch.load(cache_file)
            output_tensor = model_dict["prototype_tensor"]
            proto_labels = model_dict["proto_labels"]
            input_ids = model_dict["input_ids"]
            recalc_knn = False
            print(f"[LINE 97-101] Cache loaded. Items: {len(input_ids)}")
        else:
            print(f"[LINE 104] No cache found. Setting recalc_knn = True")
            recalc_knn = True

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    if recalc_knn:
        print(f"[LINE 108] recalc_knn is True. Initializing Dataset logic.")
        customDataSet_kw_args = {
            "eeg_data": {
                "train": "../sn2_data/organized_data/train_dict.pth",
                "train_push": "../sn2_data/organized_data/train_dict.pth",
                "eval": "../sn2_data/organized_data/train_dict.pth",
            },
            "labels": {
                "train": "../sn2_data/organized_data/sn2_train_labels.npy",
                "train_push": "../sn2_data/organized_data/sn2_train_labels.npy",
                "eval": "../sn2_data/organized_data/sn2_train_labels.npy",
            },
            "threshold": 0.5,
            "train_transform": None,
            "push_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform",
            "eval_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover",
        }

        print(f"[LINE 126] Initializing EEG_ConcatDataset...")
        knn_dataset = EEG_ConcatDataset(mode="eval", **customDataSet_kw_args)
        print(f"[LINE 126] Dataset size: {len(knn_dataset)}")

        knn_loader_config = {"batch_size": 8, "shuffle": False, "pin_memory": False}
        knn_loader = torch.utils.data.DataLoader(knn_dataset, **knn_loader_config)
        print(f"[LINE 130] DataLoader created with batch_size 8.")

        batch_embeddings = []
        input_ids = []
        proto_labels = []

        print(f"[LINE 136] Starting forward pass through training set for kNN embeddings...")
        with torch.no_grad():
            for i, sample in enumerate(knn_loader):
                inputs = sample["img"].to(device)
                input_ids += sample["sample_id"]
                proto_labels += sample["target"]

                x = model.backbone(inputs)
                x = model.add_on_layers(x).detach().cpu()

                if i == 0:
                    print(
                        f"      > First Batch Embedding Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}"
                    )

                batch_embeddings.append(x)

                if i % 100 == 0:
                    print(f"      > Processed {len(input_ids)}/{len(knn_dataset)} samples...")

        output_tensor = torch.cat(batch_embeddings, dim=0)

        print(f"[LINE 148] Total output_tensor shape: {output_tensor.shape}")
        data_dict = {
            "prototype_tensor": output_tensor,
            "proto_labels": proto_labels,
            "input_ids": input_ids,
        }

        if model_path != None:
            torch.save(data_dict, cache_file)
            print(f"[LINE 156] Saved new kNN layers to cache: {cache_file}")

    # step 2 - set the prototype layer to exactly equal the backbone output tensor
    print(f"[LINE 159] Overwriting prototype_layer.prototype_tensors with kNN data.")
    model.prototype_layer.prototype_tensors.data = output_tensor.to(device)
    proto_labels = [i.item() for i in proto_labels]
    print(f"[LINE 162] proto_labels converted to list. Size: {len(proto_labels)}")
    return model, proto_labels, input_ids


def get_demo_data():
    print(f"\n--- ENTERING get_demo_data ---")
    customDataSet_kw_args = {
        "eeg_data": {
            "train": "sample_data/sample_data.pth",
            "train_push": "sample_data/sample_data.pth",
            "eval": "sample_data/sample_data.pth",
        },
        "labels": {
            "train": "sample_data/sample_labels.npy",
            "train_push": "sample_data/sample_labels.npy",
            "eval": "sample_data/sample_labels.npy",
        },
        "threshold": 0.5,
        "train_transform": None,
        "push_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform",
        "eval_transform": f"spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover",
    }

    test_dataset = EEG_ConcatDataset(mode="eval", **customDataSet_kw_args)
    print(f"[DEMO] Dataset size: {len(test_dataset)}")

    test_loader_config = {"batch_size": 8, "shuffle": False, "pin_memory": False}
    return torch.utils.data.DataLoader(test_dataset, **test_loader_config)

def get_test_data(name):
    print(f"\n--- ENTERING get_test_data(name='{name}') ---")
    if name == "binary":
        data_dict = "test"
        test_data_name = "_test"
    elif name == "ez":
        data_dict = "test"
        test_data_name = "_ez_test"
    elif name == "val":
        data_dict = "val"      # <-- change this
        test_data_name = "_val"
    else:
        raise ValueError(f"Invalid name: '{name}'. Expected 'binary', 'ez', or 'val'.")
    
    print(f"[GET_TEST_DATA] Determined splits: data_dict={data_dict}, labels_suffix={test_data_name}")

    customDataSet_kw_args = {
        "eeg_data": {
            "train": "../sn2_data/organized_data/train_dict.pth",
            "train_push": "../sn2_data/organized_data/train_dict.pth",
            "eval": f"../sn2_data/organized_data/{data_dict}_dict.pth",  # val_dict.pth
        },
        "labels": {
            "train": "../sn2_data/organized_data/sn2_train_labels.npy",
            "train_push": "../sn2_data/organized_data/sn2_train_labels.npy",
            "eval": f"../sn2_data/organized_data/sn2{test_data_name}_labels.npy",  # sn2_val_labels.npy
        },
        "threshold": 0.5,
        "train_transform": None,
        "push_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform",
        "eval_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover",
    }

    print(f"[GET_TEST_DATA] Loading EVAL signals from: {customDataSet_kw_args['eeg_data']['eval']}")
    print(f"[GET_TEST_DATA] Loading EVAL labels from: {customDataSet_kw_args['labels']['eval']}")

    test_dataset = EEG_ConcatDataset(mode="eval", **customDataSet_kw_args)
    print(f"[GET_TEST_DATA] Final dataset loaded. Size: {len(test_dataset)}")

    test_loader_config = {"batch_size": 8, "shuffle": False, "pin_memory": False}
    return torch.utils.data.DataLoader(test_dataset, **test_loader_config)