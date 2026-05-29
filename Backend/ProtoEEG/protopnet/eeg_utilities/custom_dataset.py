import importlib
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from ..spikenet_helpers import get_all_transforms, spikenet_transform


class EEG_DataSet(Dataset):
    def __init__(
        self,
        eeg_data,
        labels,
        mode,
        threshold=0.5,
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

        self.mode = mode
        self.data = torch.load(eeg_data[self.mode], weights_only=False)
        self.labels = np.load(labels[self.mode], allow_pickle=True)
        self.threshold = threshold

        self.signal_fns = list(self.data.keys())

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

        # Binary target based on consensus threshold.
        mean_value = self.labels[index][4][0].mean()
        label = int(mean_value >= self.threshold)

        # if self.transform:
        #    img = self.transform(img)
        return {"img": img.float(), "target": label, "sample_id": sample_id}


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
        train_transform: Callable = None,
        push_transform: Callable = None,
        eval_transform: Callable = None,
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
                        print(transform_module)
                        transform_module = importlib.import_module(
                            "protopnet." + str(transform_module)
                        )
                        transform_list.append(getattr(transform_module, transform_func))
                    except Exception as e:
                        print(e)
                        raise Exception("Could not resolve transform function")

            transforms_list = [transforms.Compose(transform_list)]

        base_dataset = EEG_DataSet(
            eeg_data,
            labels,
            mode,
            threshold,
        )

        datasets = []

        for transform in transforms_list:
            # Create a new dataset that applies the transform to the samples of the base dataset

            transformed_dataset = TransformedDataset(base_dataset, transform)
            datasets.append(transformed_dataset)
            self.labels.append(transformed_dataset.labels)

        super(EEG_ConcatDataset, self).__init__(datasets)


class RandomDataset(data.Dataset):
    def __init__(self, data_size, mode):
        self.data_size = data_size
        self.mode = mode

    def __getitem__(self, index):
        # Set the random seed for generating random numbers
        torch.manual_seed(42)

        # Generate random tensor data
        random_data = torch.rand(
            ((3, 167, 20))
        )  # Example: random tensor of size [3, 32, 32]

        # Generate random tensor label
        random_label = torch.randint(0, 3, (1,))[0]

        return {
            "img": random_data,
            "target": random_label,
            "sample_id": index,
        }  # Return the random data and index in a dictionary

    def __len__(self):
        return self.data_size


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

        return {"img": signal, "sample_id": proto_id}


class SingleChannelNPDataset(Dataset):
    def __init__(
        self,
        *,
        root_dir,
        train_dir,
        train_push_dir,
        img_size,
        eval_dir=None,
        fine_annotations=False,
        fa_size=None,
        mode="train",
    ):
        super(SingleChannelNPDataset).__init__()

        self.root_dir = root_dir
        self.train_dir = train_dir
        self.train_push_dir = train_push_dir
        self.eval_dir = eval_dir
        self.img_shape = (img_size, img_size)
        self.fine_annotations = fine_annotations
        # fa_size is the dimension where fine annotations are compared to the activation pattern
        # if no fa_size is specified, use the image size (compare in pixel space)
        if fa_size:
            self.fa_shape = (fa_size, fa_size)
        else:
            self.fa_shape = self.img_shape
        self.mode = mode

        if mode == "train":
            self.root_dir = self.root_dir + self.train_dir
        elif mode == "train_push":
            self.root_dir = self.root_dir + self.train_push_dir
        elif mode == "eval":
            self.root_dir = self.root_dir + self.eval_dir
        else:
            raise ValueError(
                "mode must take the value of one of the following: train, train_push, eval"
            )

        classes, class_to_idx = self._find_classes(self.root_dir)
        self.samples = self._make_dataset(dir=self.root_dir, class_to_idx=class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample_id = path.split("/")[-1].split(".npy")[0]
        sample = np.load(path)
        sample = transforms.Compose(
            [
                torch.from_numpy,
            ]
        )(sample)

        if len(sample.shape) == 3:
            if self.fa_shape:
                resize = transforms.Resize(self.fa_shape)
                fine_anno = resize(sample[1].unsqueeze(0))
            sample = sample[0]

        if self.img_shape:
            resize = transforms.Resize(self.img_shape)
            sample = resize(sample.unsqueeze(0))

        sample = sample.expand(3, -1, -1)
        # print(f"sample shape: {sample.shape}")
        # print(f"temp shape: {s.shape}")

        # if self.transform is not None:
        #     sample = self.transform(n)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        sample_dict = {"img": sample.float(), "target": target, "sample_id": sample_id}
        if self.fine_annotations:
            sample_dict["fine_anno"] = fine_anno

        return sample_dict

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        images = []

        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
        return images


class CachedPartLabels:
    """
    Abstract base class to define the standard interface for dataset metadata handling.
    All datasets inherit common getter methods.
    """

    def __init__(self, meta_data_path: str, use_parts: bool = True) -> None:
        self.meta_data_path = meta_data_path
        self.use_parts = use_parts
        assert Path(
            self.meta_data_path
        ).exists(), f"Metadata path {meta_data_path} does not exist"
        self.cached_id_to_path = {}
        self.cached_path_to_id = {}
        self.cached_id_to_bbox = {}
        self.cached_cls_to_id = {}
        self.cached_id_to_train = {}
        self.cached_part_id_to_part = {}
        self.cached_id_to_part_centroid = {}
        self.cached_part_num = 0

        self.parse_meta_labels()
        self.check_metadata_completeness()

    def check_metadata_completeness(self):
        """Ensures that no essential metadata dictionary is empty."""
        assert self.cached_id_to_path, "id_to_path dictionary is empty"
        assert self.cached_path_to_id, "path_to_id dictionary is empty"
        assert self.cached_id_to_bbox, "id_to_bbox dictionary is empty"
        assert self.cached_cls_to_id, "cls_to_id dictionary is empty"
        assert self.cached_id_to_train, "id_to_train dictionary is empty"
        assert (
            self.cached_part_id_to_part or not self.use_parts
        ), "part_id_to_part dictionary is empty"
        assert (
            self.cached_id_to_part_centroid or not self.use_parts
        ), "id_to_part_centroid dictionary is empty"
        assert (
            self.cached_part_num > 0 or not self.use_parts
        ), "No parts are defined in part_num"

    def parse_common_meta_labels(self, cast_id_to_int=True):
        img_txt = Path(self.meta_data_path, "images.txt")
        cls_txt = Path(self.meta_data_path, "image_class_labels.txt")
        bbox_txt = Path(self.meta_data_path, "bounding_boxes.txt")
        train_txt = Path(self.meta_data_path, "train_test_split.txt")

        # id_to_path: Get the image path of each image according to its image id
        cached_id_to_path = {}
        with open(img_txt, "r") as f:
            img_lines = f.readlines()
        for img_line in img_lines:
            if cast_id_to_int:
                img_id, img_path = (
                    int(img_line.split(" ")[0]),
                    img_line.split(" ")[1][:-1],
                )
            else:
                img_id, img_path = img_line.split(" ")[0], img_line.split(" ")[1][:-1]
            img_folder, img_name = img_path.split("/")[0], img_path.split("/")[1]
            cached_id_to_path[img_id] = (img_folder, img_name)

        # id_to_bbox: Get the bounding box annotation (bird part) of each image according to its image id
        cached_id_to_bbox = {}
        with open(bbox_txt, "r") as f:
            bbox_lines = f.readlines()
        for bbox_line in bbox_lines:
            cts = bbox_line.split(" ")
            img_id, bbox_x, bbox_y, bbox_width, bbox_height = (
                int(cts[0]) if cast_id_to_int else cts[0],
                int(cts[1].split(".")[0]),
                int(cts[2].split(".")[0]),
                int(cts[3].split(".")[0]),
                int(cts[4].split(".")[0]),
            )
            bbox_x2, bbox_y2 = bbox_x + bbox_width, bbox_y + bbox_height
            cached_id_to_bbox[img_id] = (bbox_x, bbox_y, bbox_x2, bbox_y2)

        # cls_to_id: Get the image ids of each class
        cls_to_id = {}
        with open(cls_txt, "r") as f:
            cls_lines = f.readlines()
        for cls_line in cls_lines:
            img_id, cls_id = (
                (
                    int(cls_line.split(" ")[0])
                    if cast_id_to_int
                    else cls_line.split(" ")[0]
                ),
                int(cls_line.split(" ")[1]) - 1,
            )  # 0 -> 199
            if cls_id not in cls_to_id.keys():
                cls_to_id[cls_id] = []
            cls_to_id[cls_id].append(img_id)

        # id_to_train: Get the training/test label of each image according to its image id
        id_to_train = {}
        with open(train_txt, "r") as f:
            train_lines = f.readlines()
        for train_line in train_lines:
            if cast_id_to_int:
                img_id, is_train = int(train_line.split(" ")[0]), int(
                    train_line.split(" ")[1][:-1]
                )
            else:
                img_id, is_train = train_line.split(" ")[0], int(
                    train_line.split(" ")[1][:-1]
                )
            id_to_train[img_id] = is_train

        path_to_id = {"_".join(v): k for k, v in cached_id_to_path.items()}

        self.cached_id_to_path = cached_id_to_path
        self.cached_path_to_id = path_to_id
        self.cached_id_to_bbox = cached_id_to_bbox
        self.cached_cls_to_id = cls_to_id
        self.cached_id_to_train = id_to_train

    def parse_meta_labels(self):
        """
        Parses the dataset-specific metadata files. Needs to be implemented.
        """

    def id_to_path(self, id: int) -> Tuple[str, str]:
        """Return the path corresponding to a given ID."""
        return self.cached_id_to_path.get(id)

    def path_to_id(self, path: str) -> int:
        """Return the ID corresponding to a given path."""
        return self.cached_path_to_id.get(path)

    def id_to_bbox(self, id: int) -> Tuple[int, int, int, int]:
        """Return the bounding box corresponding to a given ID."""
        return self.cached_id_to_bbox.get(id)

    def cls_to_id(self, cls_id: int) -> List[int]:
        """Return the IDs corresponding to a given class."""
        return self.cached_cls_to_id.get(cls_id, [])

    def id_to_train(self, id: int) -> bool:
        """Return the binary training/test flag corresponding to a given ID."""
        return self.cached_id_to_train.get(id)

    def part_id_to_part(self, part_id: int) -> str:
        """Return the part name corresponding to a given part ID."""
        return self.cached_part_id_to_part.get(part_id)

    def id_to_part_centroid(self, id: int) -> List[Tuple[int, int, int]]:
        """Return the part locations corresponding to a given ID."""
        return self.cached_id_to_part_centroid.get(id, [])

    def get_part_num(self) -> int:
        """Return the number of parts managed by the dataset."""
        return self.cached_part_num


class ImageFolderDict(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        cached_part_labels: CachedPartLabels = None,
    ):
        super(ImageFolderDict, self).__init__(
            root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        self.cached_part_labels = cached_part_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: A dict containing a sample with key "img", the corresponding label with key "target".
                and the sample ID with key "sample_id".
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        ori_sample_wh = sample.size

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample_dict = {"img": sample, "target": target, "sample_id": str(index)}
        if self.cached_part_labels is not None:
            original_id = self.cached_part_labels.path_to_id(
                "_".join(path.split("/")[-2:])
            )
            assert (
                original_id is not None
            ), f"Original ID {original_id} from {path} not found in metadata file"

            # convert all part labels to 0-1 scale
            bbox = torch.tensor(self.cached_part_labels.id_to_bbox(original_id)).float()
            bbox[0] /= ori_sample_wh[0]
            bbox[1] /= ori_sample_wh[1]
            bbox[2] /= ori_sample_wh[0]
            bbox[3] /= ori_sample_wh[1]
            bbox = torch.clamp(bbox, max=1.0)

            if self.cached_part_labels.use_parts:
                part_centroid = torch.tensor(
                    self.cached_part_labels.id_to_part_centroid(original_id)
                ).float()
                part_centroid[:, 1] /= ori_sample_wh[0]
                part_centroid[:, 2] /= ori_sample_wh[1]
                part_centroid[:, 1:] = torch.clamp(part_centroid[:, 1:], max=1.0)

                sample_dict["sample_parts_centroids"] = part_centroid
            else:
                part_centroid = torch.tensor(
                    self.cached_part_labels.id_to_part_centroid(original_id)
                ).float()
                sample_dict["sample_parts_centroids"] = part_centroid

            sample_dict["sample_bounding_box"] = bbox

        return sample_dict


class TensorToDictDatasetAdapter(data.Dataset):
    """
    Simple adapter for a tensor dataset that relies on ordering of the returns to create
    a dictionary dataset compatible with protopnext dataset format.
    """

    def __init__(self, tensor_dataset):
        """
        Args:
            tensor_dataset (torch.utils.data.Dataset): The tensor dataset to adapt
        """
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        """
        Returns a dictionary with the sample data and target
        """
        sample = self.tensor_dataset[index]
        if hasattr(sample, "__iter__"):
            if len(sample) == 2:
                return {"img": sample[0], "target": sample[1]}
            elif len(sample) == 3:
                return {"img": sample[0], "target": sample[1], "sample_id": sample[2]}
            else:
                raise NotImplementedError("Expected sample to be length 1, 2, or 3")
        else:
            return {"img": sample}


def uneven_collate_fn(batch, stack_ignore_key):
    """
    Collates a batch of data similar to default stacking collate fn. However,
    this function zips the key entries that have uneven dimensions (number of
    elements). This is useful when the data samples have different number of values.
    For example, image samples may have different number of visible, labeled parts.

    Parameters:
    - batch (list of dicts): A batch of data where each item is a dictionary
      representing one data sample.
    - stack_ignore_key (str, optional): The key for which custom collation is to be
      bypassed. This key's data will not be stacked into a tensor.

    Returns:
    - dict: A dictionary where keys correspond to the keys in the original data
      samples, and values are the data from each sample collated. All values in
      this dict will be tensors except for the values for 'stack_ignore_key', which
      will be a list of tensors. The list will contain a tensor for each sample,
      with the first dimension of each tensor representing the index of whatever
      is being stacked.
    """

    batched_data = {}

    for key in batch[0].keys():
        batched_data[key] = []
    for item in batch:
        for key in item:
            batched_data[key].append(item[key])

    for key in batched_data:
        if all(isinstance(x, torch.Tensor) for x in batched_data[key]):
            if key != stack_ignore_key:  # We already handled 'sample_parts_centroids'
                batched_data[key] = torch.stack(batched_data[key])
        if all(isinstance(x, int) for x in batched_data[key]):
            batched_data[key] = torch.tensor(batched_data[key])
    return batched_data
