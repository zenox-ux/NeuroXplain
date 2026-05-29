from protopnet.spikenet_helpers import (
    label_finder,
    eeg_preprocess_for_model
)
import torch
import os
import numpy as np
from viz_utils import (
    local_analysis,
)
from protopnet.knn_models import ProtoEEGkNN


def get_eeg_for_model(eeg_filename, train_dict, test_dict):
    
    try:
        eeg_raw = train_dict[eeg_filename]
    except:
        eeg_raw = test_dict[eeg_filename]

    eeg = eeg_preprocess_for_model(eeg_raw).float()
    
    return eeg


save_dir = "viz"
train_dict = torch.load("../sn2_data/organized_data/train_dict.pth")
test_dict = torch.load("sample_data/sample_data.pth")

model_path = "models/trained_model_2.pth"
model = ProtoEEGkNN(model_path)

model.base_model.eval().to(model.device)
spikenet_weight_dict = torch.load("model_feats/spikenet_labels.pth")

topk = 20
count = 0
for eeg_filename in test_dict.keys():

    eeg = get_eeg_for_model(eeg_filename, test_dict, test_dict)

    output_dict = model.forward(eeg, [eeg_filename])
    proto_filenames = output_dict["matches"][0][:topk]

    assert (eeg_filename in proto_filenames) == False

    interp_path = os.path.join(save_dir, "ProtoEEGknn")
    os.makedirs(interp_path, exist_ok=True)
    local_analysis(
        model.base_model,
        eeg_filename,
        proto_filenames,
        label_finder(eeg_filename),
        train_dict,
        test_dict,
        f"{interp_path}/{eeg_filename.replace('.mat', '')}",
    )
    
    count += 1
    if count == 10: break
