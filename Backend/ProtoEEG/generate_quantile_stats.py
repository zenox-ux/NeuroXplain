import torch
import numpy as np
import pickle
import os
from tqdm import tqdm

# 1. Paths
TRAIN_DICT_PATH = "../sn2_data/organized_data/train_dict.pth"
OUTPUT_PATH = "model_feats/quantile_summary_stats.pkl"

if not os.path.exists("model_feats"):
    os.makedirs("model_feats")

print("Loading training dictionary...")
train_dict = torch.load(TRAIN_DICT_PATH)

all_ranges = []
all_vars = []

print("Calculating statistics for all training samples...")
for key in tqdm(train_dict):
    signal = train_dict[key]
    
    # We only care about the first 19/20 channels used for stats
    # match the logic in spikenet_features_summary
    eeg = signal[0:19] 
    
    # Calculate Range (Max - Min) per channel
    r = np.max(eeg, axis=1) - np.min(eeg, axis=1)
    all_ranges.extend(r.tolist())
    
    # Calculate Variance per channel
    v = np.var(eeg, axis=1)
    all_vars.extend(v.tolist())

print("Computing percentiles...")
# The code expects 100 buckets (0th to 100th percentile)
range_quantiles = np.percentile(all_ranges, np.arange(101))
var_quantiles = np.percentile(all_vars, np.arange(101))

# Match the dictionary structure expected in activations.py line 215-224
stats_dict = {
    "full": {
        "range": torch.tensor(range_quantiles).float(),
        "var": torch.tensor(var_quantiles).float()
    }
}

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(stats_dict, f)

print(f"Successfully created {OUTPUT_PATH}")