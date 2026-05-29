import torch
import numpy as np
import random
import os
import sys

def create_val_split(pth_path, npy_path, output_dir,
                     num_pos=130, num_neg=70, seed=42):

    random.seed()

    data_dict = torch.load(pth_path)
    labels_data = np.load(npy_path, allow_pickle=True)

    label_map = {}
    full_rows = {}

    for row in labels_data:
        if row is None:
            continue

        fname_arr = row[0]
        label = row[4]

        if fname_arr is None or label is None:
            continue

        fname = fname_arr[0]
        val = float(label[0][0])

        label_map[fname] = val
        full_rows[fname] = row  # store ORIGINAL row (no modification)

    # Split
    pos_files = [f for f in data_dict if f in label_map and label_map[f] == 1.0]
    neg_files = [f for f in data_dict if f in label_map and label_map[f] == 0.0]

    print(f"Available positives: {len(pos_files)}")
    print(f"Available negatives: {len(neg_files)}")

    if len(pos_files) < num_pos or len(neg_files) < num_neg:
        raise ValueError("Not enough samples to satisfy requested split")

    selected_pos = random.sample(pos_files, num_pos)
    selected_neg = random.sample(neg_files, num_neg)

    selected_files = selected_pos + selected_neg
    random.shuffle(selected_files)

    # ===== CREATE PTH (IDENTICAL FORMAT) =====
    val_dict = {}
    for fname in selected_files:
        val_dict[fname] = data_dict[fname]

    # ===== CREATE NPY (IDENTICAL FORMAT) =====
    val_labels = []

    for fname in selected_files:
        # append EXACT original row (no reconstruction)
        val_labels.append(full_rows[fname])

    val_labels = np.array(val_labels, dtype=object)

    # Save
    os.makedirs(output_dir, exist_ok=True)

    torch.save(val_dict, os.path.join(output_dir, "val_dict.pth"))
    np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)

    print("\n===== VAL SET CREATED =====")
    print(f"Total: {len(selected_files)}")
    print(f"Positives: {num_pos}")
    print(f"Negatives: {num_neg}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_val_split.py <train.pth> <train.npy> <output_dir>")
    else:
        create_val_split(sys.argv[1], sys.argv[2], sys.argv[3])