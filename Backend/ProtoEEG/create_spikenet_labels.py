import torch
import tensorflow as tf
import numpy as np
import os
import gc
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTION: PRE-PROCESSING
# -----------------------------------------------------------------------------
def leave_one_channel_in(signal):
    # signal: 1sec data chunk (expected 19 or 20 channels)
    eeg = signal[0:19]

    # average reference
    x1 = eeg - eeg.mean(axis=0)

    # bipolar montage channels
    channels = ["Fp1","F3","C3","P3","F7","T3","T5","O1","Fz","Cz","Pz","Fp2","F4","C4","P4","F8","T4","T6","O2"]
    bipolar_channels = [
        "Fp1-F7","F7-T3","T3-T5","T5-O1","Fp2-F8","F8-T4","T4-T6","T6-O2",
        "Fp1-F3","F3-C3","C3-P3","P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2",
        "Fz-Cz","Cz-Pz"
    ]

    idx = np.array([
        [channels.index(bc.split("-")[0]), channels.index(bc.split("-")[1])]
        for bc in bipolar_channels
    ])

    x2 = eeg[idx[:, 0]] - eeg[idx[:, 1]]

    # concatenate avg and bipolar, then flatten
    x = np.array([*x1, *x2])
    z1 = x.ravel()

    # zero-pad to match legacy input shape requirements
    z2 = np.zeros((36, len(z1)))
    signal = np.array([z1, *z2])
    return signal

# -----------------------------------------------------------------------------
# 2. SETUP & PATHS
# -----------------------------------------------------------------------------
# Load legacy model (Keras 2.2.2 / Python 3.7)
print("Loading SpikeNet Model...")
with open("protopnet/pretrained/model_fold_1_structure.txt", "r") as fff:
    json_string = fff.read()
model = tf.keras.models.model_from_json(json_string)
model.load_weights("protopnet/pretrained/model_fold_1_weight.h5")

# Paths
CHUNK_DIR = "model_feats/chunks"
FINAL_SAVE_PATH = "model_feats/spikenet_labels.pth"
os.makedirs(CHUNK_DIR, exist_ok=True)

# Load data dictionaries
print("Loading data dictionaries...")
train_dict = torch.load("../sn2_data/organized_data/train_dict.pth")
test_dict = torch.load("../sn2_data/organized_data/test_dict.pth")
eeg_ids = list(train_dict.keys()) + list(test_dict.keys())

# RESUME LOGIC: Check what is already finished in the chunks folder
finished_ids = set()
chunk_files = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".pth")]
if chunk_files:
    print(f"Checking existing progress in {CHUNK_DIR}...")
    for f in chunk_files:
        try:
            chunk = torch.load(os.path.join(CHUNK_DIR, f))
            finished_ids.update(chunk.keys())
        except:
            continue

pending_ids = [i for i in eeg_ids if i not in finished_ids]
print(f"Total: {len(eeg_ids)} | Finished: {len(finished_ids)} | Remaining: {len(pending_ids)}")

# -----------------------------------------------------------------------------
# 3. MAIN BATCHED PROCESSING
# -----------------------------------------------------------------------------
BATCH_SIZE = 8
current_chunk_dict = {}

if len(pending_ids) > 0:
    for i in tqdm(range(0, len(pending_ids), BATCH_SIZE), desc="Processing Batches"):
        batch_ids = pending_ids[i : i + BATCH_SIZE]
        batch_inputs = []
        valid_batch_ids = []

        for eeg_id in batch_ids:
            try:
                # Load signal
                try: eeg = train_dict[eeg_id]
                except: eeg = test_dict[eeg_id]
                
                # Handle 192-sample windows vs 128-sample windows
                if eeg.shape[1] == 192:
                    eeg = eeg[:, 32:160]
                
                # Pre-process
                signal = leave_one_channel_in(eeg)
                t_signal = torch.from_numpy(signal).unsqueeze(0)
                
                # Unfold to sub-windows [37, 128, 37]
                batched_input = t_signal.unfold(dimension=-1, size=128, step=128)
                batched_input = batched_input.squeeze(0).transpose(0, 1)
                
                # Transpose to [37, 128, 1, 37]
                batched_input = batched_input.transpose(-1, -2).unsqueeze(2)
                
                batch_inputs.append(batched_input.numpy())
                valid_batch_ids.append(eeg_id)
            except Exception as e:
                # print(f"Error processing {eeg_id}: {e}")
                continue

        if not batch_inputs:
            continue

        # Combine whole batch into one large inference call
        final_input = np.concatenate(batch_inputs, axis=0)
        
        # Predict
        predictions = model.predict(final_input, batch_size=len(final_input), verbose=0)[:, 0]
        
        # Store sub-window results back into dictionary
        for j, eeg_id in enumerate(valid_batch_ids):
            # Each ID produces 37 sub-windows
            current_chunk_dict[eeg_id] = torch.from_numpy(predictions[j*37 : (j+1)*37]).clone()

        # EVERY 1000 IDs: Save chunk to disk and WIPE RAM
        if len(current_chunk_dict) >= 1000:
            chunk_name = f"chunk_{len(finished_ids) + i}.pth"
            torch.save(current_chunk_dict, os.path.join(CHUNK_DIR, chunk_name))
            
            # THE RAM FIX: Empty dictionary and clear TensorFlow session
            current_chunk_dict = {}
            tf.keras.backend.clear_session()
            gc.collect()

    # Save final leftover batch
    if current_chunk_dict:
        torch.save(current_chunk_dict, os.path.join(CHUNK_DIR, f"chunk_final.pth"))

# -----------------------------------------------------------------------------
# 4. FINAL MERGE
# -----------------------------------------------------------------------------
print("\nMerging all chunks into one final file...")
final_labels = {}
all_chunks = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".pth")]
for f in tqdm(all_chunks, desc="Merging"):
    chunk = torch.load(os.path.join(CHUNK_DIR, f))
    final_labels.update(chunk)

# Save to the exact location required by the rest of the project
torch.save(final_labels, FINAL_SAVE_PATH)
print(f"Success! Final file saved to {FINAL_SAVE_PATH}")
print(f"Total processed samples: {len(final_labels)}")