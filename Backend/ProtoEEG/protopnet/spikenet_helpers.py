import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import importlib
import os
import re
from heapq import nlargest
import numpy as np
from collections import defaultdict


def threshold_temperature_softmax(
    logits, labels, dim=1, temperature=1.0, threshold=0.015
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Labels are expected in [0, 1] for binary use.
    labels = labels.to(device).float().clamp(0.0, 1.0)

    # Create mask for values above threshold
    mask = logits > threshold

    # Zero out values below threshold
    filtered_logits = torch.where(mask, logits, torch.zeros_like(logits))

    # Apply exponential
    exp_logits = torch.exp(filtered_logits)

    # Apply temperature scaling
    scaled_logits = exp_logits / temperature

    # Calculate sum along specified dimension, keeping dimensions
    exp_sum = torch.sum(scaled_logits * mask, dim=-1, keepdim=True)

    # Apply softmax normalization only to values that were above threshold
    result = torch.where(mask, scaled_logits / exp_sum, torch.zeros_like(logits))

    # Create uniform distribution tensor of same shape as logits
    uniform = torch.ones_like(logits) / 37

    # Reshape labels to match logits shape for broadcasting
    if len(logits.shape) == 1:
        # If logits is just [37], labels should be scalar
        labels = labels.item()
    elif len(logits.shape) == 2:
        # If logits is [n, 37], labels should be [n, 1]
        labels = labels.view(-1, 1)
    else:  # len(logits.shape) == 3
        # If logits is [n, 1, 37], labels should be [n, 1, 1]
        labels = labels.view(-1, 1, 1)

    final_result = labels * result.to(device) + (1 - labels) * uniform.to(device)
    return final_result


def topk_temperature_softmax(logits, labels, k=5, dim=1, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Labels are expected in [0, 1] for binary use.
    labels = labels.to(device).float().clamp(0.0, 1.0)

    # Get top k values and their indices
    top_values, top_indices = torch.topk(logits, k=k, dim=-1)

    # Create a mask of zeros and fill with ones at top k positions
    mask = torch.zeros_like(logits)
    mask.scatter_(-1, top_indices, 1.0)

    # Zero out all values except top k
    filtered_logits = logits * mask

    # Apply exponential
    exp_logits = torch.exp(filtered_logits)

    # Apply temperature scaling
    scaled_logits = exp_logits / temperature

    # Calculate sum along specified dimension, keeping dimensions
    exp_sum = torch.sum(scaled_logits, dim=-1, keepdim=True)

    # Apply softmax normalization only to top k values
    result = torch.where(mask > 0, scaled_logits / exp_sum, torch.zeros_like(logits))

    # Create uniform distribution tensor of same shape as logits
    uniform = torch.ones_like(logits) / 37

    # Reshape labels to match logits shape for broadcasting
    if len(logits.shape) == 1:
        # If logits is just [37], labels should be scalar
        labels = labels.item()
    elif len(logits.shape) == 2:
        # If logits is [n, 37], labels should be [n, 1]
        labels = labels.view(-1, 1)
    else:  # len(logits.shape) == 3
        # If logits is [n, 1, 37], labels should be [n, 1, 1]
        labels = labels.view(-1, 1, 1)

    final_result = labels * result.to(device) + (1 - labels) * uniform.to(device)
    return final_result


def select_best_models_given_same_name(file_list, path):
    # Dictionary to store the best file for each model ID
    best_models = defaultdict(lambda: ("", float("-inf")))

    # Regular expression to extract model ID and number
    pattern = (
        r"./live/artifacts/" + str(path) + "/([^/]+)/(\d+)_project_(\d+\.\d+)\.pth"
    )

    for file_path in file_list:
        match = re.match(pattern, file_path)
        if match:
            model_id, epoch, number = match.groups()
            number = float(number)

            # Update if this is the best (highest number) for this model ID
            if number > best_models[model_id][1]:
                best_models[model_id] = (file_path, number)

    # Return only the file paths of the best models
    return [file_path for file_path, _ in best_models.values()]


def find_largest_files(path):

    def extract_value(filename):
        return float(filename.split("_")[-1].split(".")[0])

    base_path = os.path.join("./live/artifacts", path)
    file_values = []
    try:
        for filename in os.listdir(base_path):
            if filename.endswith(".pth"):
                match = re.search(r"_(\d+(?:\.\d+)?)\.pth$", filename)
                if match:
                    float(match.group(1))
                    file_values.append((filename))

        largest_files = nlargest(20, file_values, key=extract_value)

        return [(os.path.join(base_path, filename)) for filename in largest_files]
    except FileNotFoundError:
        print(f"Error: Directory '{base_path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied to access directory '{base_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return []


def eeg_crop(x):
    """
    x: input to be transformed

    returns: center cropped version of x.
    Meant to be used before spikenet_transform is applied
    """
    crop = transforms.CenterCrop((20, 128))
    x = crop(x)

    return x


def spikenet_transform(x):
    """
    This function created spikenet data by concatenating AVG and L-bipolar data formats

    Parameters:
    -----------
        x (torch.Tensor): The input data to be transformed.

    Returns:
    --------
        torch.Tensor: The transformed data in the Spikenet format.

    Notes:
    ------
        The input data should be a torch.Tensor of shape (C, T), where C represents the number of channels and T represents the number of time steps.
        The Spikenet data format is created by concatenating the AVG (average) and L-bipolar data formats.

    Examples:
    ---------
        input_data = torch.randn(19, 1000)
        transformed_data = spikenet_transform(input_data)

    """

    # indices: list of pairs where list[0]-list[1] is l2bipolar
    bp_indices = [
        [0, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [11, 15],
        [15, 16],
        [16, 17],
        [17, 18],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 7],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 18],
        [8, 9],
        [9, 10],
    ]

    x = x[:-1]  # take out last row (ekg)
    avg = x - x.mean(axis=0)

    bipolar = torch.clone(x)
    for pair in range(len(bp_indices)):
        i, j = bp_indices[pair]
        bipolar[pair] = x[i] - x[j]

    bipolar = bipolar[:-1]

    return torch.cat((avg, bipolar), axis=0).transpose(0, 1).unsqueeze(0)


def extremes_remover(signal, signal_min=0.001, signal_max=2000, verbose=True):
    """
    Zeros out channels outside amplitude range.
    FIXED: Default min is 0.001 because NMT data is normalized/scaled.
    """
    # signal shape: [1, 128, 37]
    total_channels = signal.shape[2]
    zeroed_count = 0

    for channel in range(total_channels):
        if signal[:, :, channel].numel() > 0:
            pp = signal[:, :, channel].max() - signal[:, :, channel].min()
            
            # If PP is smaller than 0.001, it's a dead channel or flat line
            if (pp < signal_min) or (pp > signal_max):
                signal[:, :, channel] = 0
                zeroed_count += 1
        else:
            signal[:, :] = 0
            zeroed_count = total_channels
            break

    # Diagnostic print to confirm the fix is working
    if verbose and zeroed_count > 0:
        if not hasattr(extremes_remover, "count"): extremes_remover.count = 0
        if extremes_remover.count < 5: # Only show for first few samples
            print(f"[DEBUG] extremes_remover: Zeroed {zeroed_count}/{total_channels} channels. (Sample PP was {pp:.4f})")
            if zeroed_count == total_channels:
                print("  ! WARNING: ALL channels zeroed. Check if signal_min is too high.")
            extremes_remover.count += 1
            
    return signal

def normalizer(signal):
    if signal.shape[-1] != 0:
        # normalize signal
        signal = signal / (
            np.quantile(np.abs(signal), q=0.95, method="linear", axis=1, keepdims=True)
            + 1e-8
        )

    return signal


def eeg_preprocess_for_plotting(eeg):
    """
    preprocess the EEG for appropriate cropping, channels, and transpose
    *USED ONLY FOR PLOTTING AND VISUALIZATION*

    """

    preprocess = transforms.Compose([eeg_crop, spikenet_transform])

    return preprocess(torch.from_numpy(eeg))[0].T


def eeg_preprocess_for_model(eeg):
    """
    preprocess the EEG for appropriate cropping, channels, and transpose
    *USED ONLY FOR forward prop 1 value through the model*

    Recall - the model takes inputs of shape [bsz, 1, 128, 37]

    """

    preprocess = transforms.Compose([eeg_crop, spikenet_transform])

    return preprocess(torch.from_numpy(eeg)).unsqueeze(0).float()


def get_all_transforms():
    """
    Returns a list of all possible combinations of transformations.

    The returned list, `all_transforms`, contains combinations of shift, flip, and jitter transforms that can be applied to data.
    Each combination consists of a sequence of transformations applied in order.

    **TRANSFORMS are either applied with crop then spikenet or one transform from all_transforms then

    Returns:
        list: A list of `torchvision.transforms.Compose` objects representing different combinations of transformations.
    """

    crop = transforms.CenterCrop((20, 128))

    shift = transforms.CenterCrop((20, 130))
    shift_L = transforms.Lambda(lambda x: shift(x)[:, 2:])
    shift_R = transforms.Lambda(lambda x: shift(x)[:, :-2])

    # flip the data
    # indices 8,9,10,19 are left untouched indices [0-7] (inclusive) are now [11-18] and vice-versa
    flip = transforms.Lambda(
        lambda x: torch.cat((x[11:19], x[8:11], x[:8], x[19].unsqueeze(0)))
    )

    # multiple all amplitudes by (1+eps) where eps is gaussian(0,1)
    jitter_amp = transforms.Lambda(
        lambda x: x
        * (
            1
            + torch.normal(
                mean=torch.zeros((x.shape[0], x.shape[1])),
                std=0.1 * torch.ones((x.shape[0], x.shape[1])),
            )
        )
    )

    # 12 combinations -
    all_transforms = []

    shifts = [crop, shift_R, shift_L]
    flips = [flip, None]
    jitters = [jitter_amp, None]

    for i in shifts:
        for j in flips:
            for k in jitters:
                t = [i]
                if j is not None:
                    t.append(j)
                if k is not None:
                    t.append(k)

                t.append(spikenet_transform)
                all_transforms.append(transforms.Compose(t))

    return all_transforms


class ReshapeLayer(nn.Module):
    def __init__(self, new_shape):
        super(ReshapeLayer, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(self.new_shape)


class EEG_DataSet(Dataset):
    def __init__(
        self,
        eeg_data,
        labels,
        threshold=0.5,
        transform=None,
    ):
        """
        Each data, i, contains a file name at index 0 and the number of votes at index 4

        Parameters:
        ----------
            eeg_data (str): The file path containing the EEG recordings.
            labels (str): The file path containing the corresponding labels.
            threshold (float, optional): The threshold value used to determine the label.
                Defaults to 0.5.
            transform (callable, optional): Optional data transformation to be applied.
                It should be a callable that takes in a sample and returns the transformed sample.
                Defaults to None.
            mode (str, optional): The mode of the dataset. Can be either 'train', 'train_push', or 'eval'.

        Notes:
        ------
            The data and labels files should be saved using `torch.save()` and have the following format:
            - data: a dictionary where each key represents a sample and its value is a numpy array of the EEG recording.
            - labels: a list of labels corresponding to each sample, where each label is a list containing the file name and number of votes.

        Examples:
        ---------
            dataset = EEG_DataSet(eeg_data='data.pth', labels='labels.pth', threshold=0.5, transform=transforms.ToTensor())
            sample = dataset[0]

        """
        self.data = torch.load(eeg_data)
        self.labels = np.load(labels, allow_pickle=True)
        self.threshold = threshold
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        --------
            int: The total number of samples.

        """
        return self.labels.shape[0]

    def __getitem__(self, index):
        """
        Returns a single sample from the dataset at the given index.

        Parameters:
        -----------
            index (int): The index of the sample to retrieve.

        Returns:
        --------
            dict: A dict containing the EEG recording with key "img", the corresponding label with key "target".
                and the sample ID with key "sample_id".
                The EEG recording is a torch.Tensor of shape (C, H, W), and the label is an int.

        """
        sample_id = self.labels[index][0][0]

        img = torch.from_numpy(self.data[self.labels[index][0][0]])

        mean_value = self.labels[index][4][0].mean()
        label = int(mean_value >= self.threshold)

        if self.transform:
            img = self.transform(img)

        return {"img": img.float(), "target": label, "sample_id": sample_id}


# REPASTED FROM EEG_UTILITES.CUSTOM_DATASET


class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        self.labels = self.base_dataset.labels

    def __getitem__(self, index):
        x = self.base_dataset[index]["img"]
        y = self.base_dataset[index]["target"]
        sample_id = self.base_dataset[index]["sample_id"]

        return {"img": self.transform(x).float(), "target": y, "sample_id": sample_id}

    def __len__(self):
        return len(self.base_dataset)


class EEG_ConcatDataset(ConcatDataset):
    def __init__(
        self,
        eeg_data,
        labels,
        mode,
        threshold=0.5,
        train_transform=None,
        push_transform=None,
        eval_transform=None,
    ):
        """
        This class represents a dataset that concatenates multiple EEG datasets with different transformations.

        Parameters:
        ----------
            eeg_data (str): The file path containing the EEG recordings.
            labels (str): The file path containing the corresponding labels.
            threshold (float, optional): The threshold value used to determine the label.
                Defaults to 0.5.
            transform (callable, optional): Optional data transformation to be applied.
                It should be a callable that takes in a sample and returns the transformed sample.
                Defaults to None.
            mode (str, optional): The mode of the dataset. Can be either 'train', 'train_push', or 'eval'.
        """
        self.mode = mode
        self.labels = []

        # filter if file has 1 channel above the threshold
        if self.mode == "train":
            self.transform = train_transform
            transforms_list = get_all_transforms()
        elif self.mode == "train_push":
            self.transform = push_transform
        elif self.mode == "eval":
            self.transform = eval_transform

        if self.mode == "train_push" or self.mode == "eval":
            transform_list = []
            if type(self.transform) is str:
                my_transforms = self.transform.split(" ")
                for transform in my_transforms:
                    try:
                        # Resolve the transform function
                        transform_module, transform_func = transform.split(".")
                        transform_module = importlib.import_module(
                            "protopnet." + str(transform_module)
                        )
                        transform_list.append(getattr(transform_module, transform_func))
                    except Exception as e:
                        print(e)
                        raise Exception("Could not resolve transform function")

            transforms_list = [transforms.Compose(transform_list)]

        base_dataset = EEG_DataSet(
            eeg_data[self.mode],
            labels[self.mode],
        )

        datasets = []

        for transform in transforms_list:
            # Create a new dataset that applies the transform to the samples of the base dataset

            transformed_dataset = TransformedDataset(base_dataset, transform)
            datasets.append(transformed_dataset)
            self.labels.append(transformed_dataset.labels)

        super(EEG_ConcatDataset, self).__init__(datasets)


class EEGProtoDataset(Dataset):
    def __init__(self, root_dir, proto_ids):
        super(EEGProtoDataset).__init__()

        self.dict = torch.load(root_dir)
        self.root_dir = root_dir
        self.proto_ids = proto_ids

    def __len__(self):
        return len(self.proto_ids)

    def __getitem__(self, idx):
        crop = transforms.CenterCrop((20, 128))
        proto_id = self.proto_ids[idx]  # name of file
        signal = torch.from_numpy(self.dict[proto_id]).float()
        signal = spikenet_transform(crop(signal))

        return {"img": signal, "target": None, "sample_id": proto_id}


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for label in loader:
            self.labels_list.append(label["target"])
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


#############################################################################################################################################################################################################################################
# SOME GLOBAL VARIABLES FOR PLOTTING
##############################################################################################################################################################################################################################################################
zscale = 1.5
Fs = 128
offset = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    11,
    12,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    23,
    24,
    25,
    26,
    28,
    29,
    30,
    31,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    41,
    43,
    44,
]
t = np.linspace(0, 1, Fs)

channel_names = [
    "Fp1-Avg",
    "F3-Avg",
    "C3-Avg",
    "P3-Avg",
    "F7-Avg",
    "T3-Avg",
    "T5-Avg",
    "O1-Avg",
    "Fz-Avg",
    "Cz-Avg",
    "Pz-Avg",
    "Fp2-Avg",
    "F4-Avg",
    "C4-Avg",
    "P4-Avg",
    "F8-Avg",
    "T4-Avg",
    "T6-Avg",
    "O2-Avg",
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fz-Cz",
    "Cz-Pz",
]


def label_finder(eeg_name):
    """
    args:
        Given any eeg file, return the label of the file

    return:
        label associated with the eeg_file


    """

    x = np.load("../sn2_data/organized_data/sn2_test_labels.npy", allow_pickle=True)
    y = np.load("../sn2_data/organized_data/sn2_train_labels.npy", allow_pickle=True)
    z = np.load("../sn2_data/organized_data/sn2_val_labels.npy", allow_pickle=True)

    all_data = np.concatenate((x, y, z), axis=0)

    for i in all_data:
        if i[0][0] == eeg_name:
            return str(round(np.sum(i[4][0]) / len(i[4][0]), 3))

    return "LABEL NOT FOUND"
