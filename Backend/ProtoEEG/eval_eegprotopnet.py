from sklearn.metrics import roc_auc_score
from protopnet.knn_models import ProtoEEGkNN
from protopnet.eval_utils import get_test_data, bootstrap_metrics_ci, get_demo_data
import torch
import argparse
import numpy as np
import time

print("[LINE 1-10] Imports completed.")
parser = argparse.ArgumentParser()
print("[LINE 11] parser = argparse.ArgumentParser()")
parser.add_argument("-path", help="path name in ./live/artifacts/")
print("[LINE 12] parser.add_argument('-path', ...)")
parser.add_argument("-topk", type=int, default=10, help="topk value to use")
print("[LINE 13] parser.add_argument('-topk', ...)")
args = parser.parse_args()
print(f"[LINE 14] args = parser.parse_args() | values: path={args.path}, topk={args.topk}")

print(f"[LINE 16] Initializing model: model = ProtoEEGkNN('{args.path}', topk={args.topk})")
model = ProtoEEGkNN(args.path, topk=args.topk)
print(f"[LINE 16] Model loaded.")

# model.prototype_layer.importance_by_statistic.data = nn.Parameter(torch.log(torch.tensor([0.000001, 0.18, 0.18, 0.64], dtype=torch.float32))).cuda()
print("[LINE 18] Commented line: model.prototype_layer.importance_by_statistic.data = ... (skipped)")

sm = torch.nn.Softmax(dim=0)
print(f"[LINE 19] sm = torch.nn.Softmax(dim=0)")

importance_stats = sm(model.base_model.prototype_layer.importance_by_statistic)
print(f"[LINE 20] importance_stats = sm(model.base_model.prototype_layer.importance_by_statistic)")
print("Model importance stats (latent, range, var, FFT): ", importance_stats)
print(f"      > Variable values: {importance_stats.detach().cpu().numpy()}")

test_loader = get_test_data("val")
print(f"[LINE 23] test_loader = get_test_data('val') | Number of batches: {len(test_loader)}")
#test_loader = get_demo_data()
print("[LINE 24] Commented line: test_loader = get_demo_data() (skipped)")

y_true = np.array([])
y_pred = np.array([])
sample_names = []
nbrs = []
sample_dict = {}
print(f"[LINE 27-31] Tracking arrays initialized: y_true, y_pred, sample_names, nbrs, sample_dict")

batch_idx = 0
for sample in test_loader:
    batch_idx += 1
    print(f"\n--- Processing Batch {batch_idx} ---")
    
    with torch.no_grad():
        eeg = sample["img"]
        input_ids = sample["sample_id"]
        target = sample["target"]
        sample_names += input_ids
        print(f"[LINE 37-40] Data Unpacked:")
        print(f"      > eeg shape: {list(eeg.shape)} | eeg mean: {eeg.mean().item():.4f}")
        print(f"      > input_ids: {input_ids}")
        print(f"      > target labels: {target.cpu().numpy()}")
        print(f"      > sample_names updated, current length: {len(sample_names)}")

        y_true = np.concatenate((y_true, target))
        print(f"[LINE 42] y_true = np.concatenate(...) | Current total size: {len(y_true)}")

        print(f"[LINE 44] Running model forward pass: output_dict = model.forward(eeg, input_ids)")
        output_dict = model.forward(eeg, input_ids)

        prediction = output_dict["prediction"]
        print(f"[LINE 46] prediction = output_dict['prediction']")
        print(f"      > values: {prediction}")

        y_pred = np.concatenate((y_pred, prediction))
        print(f"[LINE 48] y_pred = np.concatenate(...) | Current total size: {len(y_pred)}")

        # nbrs += output_dict['neighbor_labels']
        print("[LINE 50] Commented line: nbrs += output_dict['neighbor_labels'] (skipped)")
        # print(torch.abs(eeg).mean())
        print("[LINE 51] Commented line: print(torch.abs(eeg).mean()) (skipped)")

        for i in range(len(input_ids)):
            sample_dict[input_ids[i]] = {
                "label": target[i].item(),
                "prediction": prediction[i],
            }
        print(f"[LINE 54-58] sample_dict updated for batch {batch_idx}, current size: {len(sample_dict)}")

print("\n--- All batches complete. Calculating Metrics ---")
print(f"[LINE 61] Final arrays summary:")
print(f"      > y_true length: {len(y_true)} | sample: {y_true[:5]}...")
print(f"      > y_pred length: {len(y_pred)} | sample: {y_pred[:5]}...")
print(f"      > sample_names length: {len(sample_names)}")
print(f"      > nbrs length: {len(nbrs)} (empty, was commented out)")
print(f"      > sample_dict size: {len(sample_dict)}")

print(f"[LINE 63] results = bootstrap_metrics_ci(y_true, y_pred)")
results = bootstrap_metrics_ci(y_true, y_pred)

print("=" * 50)
print("Bootstrap Metrics with 95% Confidence Intervals")
print("=" * 50)
print(f"R² Score:    {results['r2']:.4f} (CI: [{results['r2_ci'][0]:.4f}, {results['r2_ci'][1]:.4f}])")
print(f"Accuracy:    {results['accuracy']:.4f} (CI: [{results['accuracy_ci'][0]:.4f}, {results['accuracy_ci'][1]:.4f}])")
print(f"AUROC:       {results['auroc']:.4f} (CI: [{results['auroc_ci'][0]:.4f}, {results['auroc_ci'][1]:.4f}])")
print("=" * 50)