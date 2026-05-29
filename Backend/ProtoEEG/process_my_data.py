import pandas as pd
import numpy as np
import scipy.io
import torch
import os
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION (CHECK THESE PATHS!)
# ==========================================
CSV_PATH = "sn2_split.csv" 
MAT_DIR = "./hardmine"      # Folder where your .mat files are
OUTPUT_DIR = "../sn2_data/organized_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD CSV (FIXED DELIMITER)
# ==========================================
# We use sep=None and engine='python' to let pandas auto-detect if it's a comma or tab
df = pd.read_csv(CSV_PATH, sep=None, engine='python')

print("CSV Columns found:", df.columns.tolist())

# Check for common naming issues (like trailing spaces)
df.columns = df.columns.str.strip()

train_dict = {}
test_dict = {}
train_labels = []
test_labels = []

# ==========================================
# 3. PROCESSING LOOP
# ==========================================
print("Building dictionaries and label arrays...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    # Using row.get() to avoid hard KeyError crashes
    event_id = str(row['event_file'])
    split = str(row['Split']).lower().strip()
    fraction = float(row['fraction_of_yes'])
    
    mat_filename = f"{event_id}.mat"
    mat_path = os.path.join(MAT_DIR, mat_filename)
    
    if os.path.exists(mat_path):
        try:
            # Load the EEG signal
            mat_data = scipy.io.loadmat(mat_path)
            signal = mat_data['data'] 
            
            # Keep label format as 2D array, but binary-oriented single value.
            votes = np.array([[fraction]])
            
            # Create the complex object array row expected by custom_dataset.py
            # Index 0: Filename, Index 4: Votes
            label_row = [np.array([mat_filename]), None, None, None, votes, None, None, None]
            
            if split == 'train':
                train_dict[mat_filename] = signal
                train_labels.append(label_row)
            else:
                test_dict[mat_filename] = signal
                test_labels.append(label_row)
        except Exception as e:
            print(f"Error loading {mat_filename}: {e}")
    else:
        # Optional: uncomment to see missing files
        # print(f"Warning: {mat_path} not found.")
        pass

# ==========================================
# 4. SAVE OUTPUTS
# ==========================================
print(f"\nFinal Statistics:")
print(f"Train samples: {len(train_dict)}")
print(f"Test samples: {len(test_dict)}")

if len(train_dict) > 0:
    torch.save(train_dict, os.path.join(OUTPUT_DIR, "train_dict.pth"))
    np.save(os.path.join(OUTPUT_DIR, "sn2_train_labels.npy"), np.array(train_labels, dtype=object))

if len(test_dict) > 0:
    torch.save(test_dict, os.path.join(OUTPUT_DIR, "test_dict.pth"))
    np.save(os.path.join(OUTPUT_DIR, "sn2_test_labels.npy"), np.array(test_labels, dtype=object))

print(f"\nSuccess! Files saved to {OUTPUT_DIR}")
