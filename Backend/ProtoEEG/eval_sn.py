#### RUN FROM PROTOPNET DIR USING 'CONDA ACTIVATE SPIKE' ####

### USE https://colab.research.google.com/drive/1xIzp7K6SAuUp7IJMcfTxdIIQxjnMDSuX#scrollTo=9zifJuBhN3zL for bootstrapping ETC ###
import argparse
import torch
import tensorflow as tf
import numpy as np
from protopnet.eval_utils import get_test_data, get_SAI_data
from protopnet.hm_utils import get_mAUC_data
import os

parser = argparse.ArgumentParser(
    description="Find the 5 largest files in a specific artifact path."
)
parser.add_argument(
    "-test_data",
    choices=["binary", "ez", "mAUC", "SAI"],
    help="data type to use ('binary' or 'ez')",
    default="binary",
)
parser.add_argument(
    "-chunk",
    type=int,
    choices=range(65),
    help="integer value from 0-19 inclusive",
    default=0,
)
args = parser.parse_args()

# load in the model weights (must use python 3.7 and keras 2.2.2)
with open("protopnet/pretrained/model_fold_1_structure.txt", "r") as fff:
    json_string = fff.read()
model = tf.keras.models.model_from_json(json_string)
model.load_weights("protopnet/pretrained/model_fold_1_weight.h5")

if args.test_data == "mAUC":
    chunk = args.chunk
    test_loader = get_mAUC_data(chunk)

elif args.test_data == "SAI":

    files = os.listdir("./model_feats/SAI/")
    sai_choices = []
    for i in files:
        if "SAI" in i:
            sai_choices.append(i)

    sai_snweights_file = sai_choices[args.chunk]

    SAI_filename = sai_snweights_file.replace("sn_channelweight_", "")
    SAI_path = f"../external_data/processed/{SAI_filename}"

    test_loader = get_SAI_data(SAI_path)

else:
    test_loader = get_test_data(args.test_data)

print(len(test_loader))

y_true = np.array([])
y_pred = np.array([])

labels = []
predictions = []

bce = []
pred_dict = {}
for sample in test_loader:

    # Spikenet takes in transposed inputs of 128, 37
    input = sample["img"].transpose(1, 2).numpy()
    input = tf.convert_to_tensor(input)
    samples = sample["sample_id"]
    labels += sample["target"]

    # get the predictions from the model and flatten them, add to y_pred
    outputs = model.predict(input)
    y_pred = list(outputs.flatten())

    for eeg, pred in zip(samples, y_pred):
        pred_dict[eeg] = pred

    predictions += y_pred


if args.test_data == "binary":
    save_loc = f"NEJM/eval/spikenet/binary.pth"
if args.test_data == "ez":
    save_loc = f"NEJM/eval/spikenet/binarized.pth"
if args.test_data == "mAUC":
    save_loc = f"NEJM/eval/spikenet/mAUC_{chunk}.pth"
if args.test_data == "SAI":
    save_loc = f"NEJM/eval/spikenet/{SAI_filename}"

torch.save(pred_dict, save_loc)
