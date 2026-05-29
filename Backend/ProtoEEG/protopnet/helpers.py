import gc
import hashlib
import io
import itertools
import json
import operator
import os
import subprocess
import sys
from typing import Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def submit_full_pacmap_run(dataloader, model, save_dir, image_name, binary=False):
    # edit the save dir in here
    save_dir = os.path.join(save_dir, image_name)
    os.makedirs(save_dir, exist_ok=True)

    # save model as model.pth
    torch.save(model, os.path.join(save_dir, "model.pth"))  # maybe make a specific name

    def generate_sh(save_dir):
        form = f"""#!/bin/bash 
#SBATCH --job-name={save_dir[-8:]}
#SBATCH -t 60:00:00  # time requested in hour:minute:second 
#SBATCH -o {save_dir}/pmpOut 
#SBATCH -e {save_dir}/pmpErr
#SBATCH --mem=60G
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A rudin
#SBATCH -p rudin

source /home/users/dt161/miniconda3/etc/profile.d/conda.sh
conda activate protopnext
python3 gen_pacmap_from_training.py -save_dir {save_dir}"""
        return form.format(save_dir)

    # create .sh file
    sh_file_location = os.path.join(save_dir, "generate_pacmap.sh")

    with open(sh_file_location, "w") as f:
        f.write(generate_sh(save_dir))

    # submit the .sh command
    command = f"sbatch {sh_file_location}"
    subprocess.Popen(
        command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def hash_func(img: torch.tensor):
    """
    Takes in a tensor, outputs hash of that tensor as string.
    """
    buffer = io.BytesIO()
    torch.save(img, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def indices_to_upsampled_boxes(indices, latent_size, image_size, align_corners=True):
    """
    Maps indices from (-1, 1) to the size of the image
    """
    # Mapped to 0, 1
    indices = (indices + 1) / 2

    if align_corners:
        box_shape = (image_size[0] / latent_size[0], image_size[1] / latent_size[1])
        reduced_image_size = (
            image_size[0] - box_shape[0],
            image_size[1] - box_shape[1],
        )

        box_tl = (
            int(indices[0] * reduced_image_size[0]),
            int(indices[1] * reduced_image_size[1]),
        )
        box_br = (
            int(indices[0] * reduced_image_size[0] + box_shape[0]),
            int(indices[1] * reduced_image_size[1] + box_shape[1]),
        )
    else:
        box_shape = (
            image_size[0] / (latent_size[0] - 1),
            image_size[1] / (latent_size[1] - 1),
        )
        box_tl = (
            int(indices[0] * image_size[0] - box_shape[0] / 2),
            int(indices[1] * image_size[1] - box_shape[1] / 2),
        )
        box_br = (
            int(indices[0] * image_size[0] + box_shape[0] / 2),
            int(indices[1] * image_size[1] + box_shape[1] / 2),
        )

    return box_tl, box_br


def json_load(filepath):
    ret_dict = dict()
    dict_ = json.load(open(filepath, "r"))
    for k, v in dict_.items():
        ret_dict[int(k)] = v

    del dict_
    return ret_dict


def list_of_distances(X, Y):
    """
    Computes the squared Euclidean distance between each pair of vectors in X and Y.

    Parameters:
    ----------
        X (torch.Tensor): A tensor of shape (N, D) containing N D-dimensional vectors.
        Y (torch.Tensor): A tensor of shape (M, D) containing M D-dimensional vectors.

    Returns:
    --------
        torch.Tensor: A tensor of shape (N, M) containing the squared Euclidean distance between each pair of
        vectors in X and Y. The element (i, j) of the returned tensor is the squared Euclidean distance between
        the i-th vector in X and the j-th vector in Y.

    Raises:
    -------
        ValueError: If the second dimension of X and Y are not the same.

    Called in the following files:
        - Original ProtoPNet Repo: Used for p_avg_pair_dist = torch.mean(list_of_distances(p, p)) where p is the prototype_vectors
    """
    return torch.sum(
        (torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1
    )


def make_one_hot(target, target_one_hot):
    """
    Converts a tensor of target labels into a one-hot encoded tensor.

    Parameters:
    -----------
        target (torch.Tensor): A tensor of shape (N,) containing N target labels.
        target_one_hot (torch.Tensor): A tensor of shape (N, K) where K is the number of classes, initialized
        to all zeros.

    Returns:
    --------
        None. This function operates in-place and modifies the input tensor target_one_hot.

    Raises:
    -------
        ValueError: If the second dimension of target_one_hot does not match the number of unique values in target.


    Called in the following files:
        - train_and_eval.py (Imported but never used)
    """
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.0)


def makedir(path):
    """
    Create a directory at the specified path if it does not already exist.

    Parameters:
    -----------
        path (str): The path to the directory to create.

    Returns:
    --------
        None.

    Raises:
    -------
        OSError: If the directory could not be created.

    Called in the following files:
        - find_nearest.py: find_k_nearest_patches_to_prototypes()
        - global_analysis.py: save_def_prototype_patches()
        - local_analysis.py
        - main.py
        - push.py: push_prototypes()
    """
    if not os.path.exists(path):
        os.makedirs(path)


def find_high_activation_crop(activation_map, percentile=95):
    """
    Given an activation map, find the rectangular crop that contains the top `percentile` percent of activations.

    Parameters:
    -----------
        activation_map (np.ndarray): A 2D array of activation values.
        percentile (float): The percentile of activations to include in the crop. Defaults to 95.

    Returns:
    --------
        A tuple of integers (lower_y, upper_y, lower_x, upper_x), representing the coordinates of the rectangular crop
        that contains the top `percentile` percent of activations. `lower_y` and `upper_y` are the indices of the top and
        bottom rows of the crop, respectively, and `lower_x` and `upper_x` are the indices of the left and right columns
        of the crop, respectively.

    Raises:
    -------
        ValueError: If `percentile` is not between 0 and 100, or if `activation_map` is not a 2D array.

    Called in the following files:
        - find_nearest.py (not used)
        - local_analysis.py
        - push.py: update_prototypes_on_batch(), save_projected_prototype_images()
    """
    # if not isinstance(activation_map, np.ndarray) or activation_map.ndim != 2:
    #     raise ValueError("`activation_map` must be a 2D numpy array")
    # if not 0 <= percentile <= 100:
    #     raise ValueError("`percentile` must be between 0 and 100")

    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def plot_losses(args=None, directory=None):
    """
    Plots the train and validation losses from a log file and saves the plot as an image.

    Parameters:
    -----------
        input_path (str): Path to the log file containing the loss values. Default is "/sbgenomics/workspace/interpnn2023/save_path_test/saved_models/spikenet/train_dict/003/train.log".
        output_path (str): Path to save the loss plots. Default is "/sbgenomics/workspace/interpnn2023/save_path_test/saved_models/spikenet/loss_plots".

    Returns:
    --------
        None
    """
    log_file = os.path.join(directory, "train.log")
    with open(log_file, "r") as file:
        # Read the contents of the file
        contents = file.readlines()

    # Print the contents of the file
    losses = [
        "cross ent",
        "cluster",
        "separation",
        "avg separation",
        "accu",
        "orthogonality loss",
        "total loss",
        "fine annotation",
        "l1",
        "avg l2",
        "max offset",
        "orthogonality loss with weight",
    ]

    train_losses = {i: [] for i in losses}
    val_losses = {i: [] for i in losses}

    section = ""
    for i in range(len(contents)):
        curr_line = contents[i].strip()

        if curr_line == "train":
            section = "train"
            continue

        if curr_line == "val":
            section = "val"
            continue

        if section == "train":
            try:
                loss, val = contents[i].split(":")
                train_losses[loss.strip(" \t\n")].append(float(val.strip(" \t\n%")))
            except (KeyError, ValueError):
                continue
        elif section == "val":
            try:
                loss, val = contents[i].split(":")
                val_losses[loss.strip(" \t\n")].append(float(val.strip(" \t\n%")))
            except (KeyError, ValueError):
                continue

    # Create the subplots
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))

    # Customize the subplots
    counter = 0

    keys = list(train_losses.keys())
    for i in range(4):
        for j in range(3):
            axes[i, j].plot(
                np.arange(len(train_losses[keys[counter]])),
                train_losses[keys[counter]],
                label="train",
            )
            axes[i, j].plot(
                np.arange(len(val_losses[keys[counter]])),
                val_losses[keys[counter]],
                label="val",
            )
            axes[i, j].set_title(keys[counter])
            axes[i, j].legend(loc="lower right")
            counter += 1  # Example plot for demonstration purposes

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    output_path = os.path.join(directory, "loss_plot")
    plt.savefig(output_path)


def imsave_with_bbox(
    fname,
    img_rgb,
    bbox_height_start,
    bbox_height_end,
    bbox_width_start,
    bbox_width_end,
    color=(0, 255, 255),
):
    """
    Saves an RGB image with a bounding box drawn around a specified region of interest (color overlay).

    Parameters:
    -----------
        fname (str): The file name to save the image to with the bounding box.
        img_rgb (numpy.ndarray): The input image to save with the bounding box. A 3D numpy array of shape (height, width, 3) representing an RGB image.
        bbox_height_start (int): The starting height coordinate of the bounding box (top y-coordinate).
        bbox_height_end (int): The ending height coordinate of the bounding box (bottom y-coordinate).
        bbox_width_start (int): The starting width coordinate of the bounding box (left x-coordinate).
        bbox_width_end (int): The ending width coordinate of the bounding box (right x-coordinate).
        color (tuple): The RGB color to overlay on the bounding box. Default is (0, 255, 255).
            A tuple of 3 integers specifying the RGB color of the bounding box. Defaults to (0, 255, 255) which corresponds to yellow.

    Raises:
    -------
        TypeError: If the input image is not a numpy array.
        ValueError: If the input image has less than three dimensions.
        TypeError: If the input image is not in RGB format.
        ValueError: If the bounding box coordinates are invalid.

    Examples:
    ---------
    >>> img = np.ones((100,100,3), dtype=np.uint8) * 255
    >>> imsave_with_bbox("example.png", img, 25, 75, 25, 75, color=(0,0,255))
    # saves an image with a red bounding box around the central 50x50 region


    Called in the following files:
        - local_analysis.py
        - find_nearest.py: find_k_nearest_patches_to_prototypes()
    """
    try:
        img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
        cv2.rectangle(
            img_bgr_uint8,
            (bbox_width_start, bbox_height_start),
            (bbox_width_end - 1, bbox_height_end - 1),
            color,
            thickness=2,
        )
        img_rgb_uint8 = img_bgr_uint8[..., ::-1]
        img_rgb_float = np.float32(img_rgb_uint8) / 255
        plt.imsave(fname, img_rgb_float)

    except TypeError as e:
        raise TypeError("Input image must be a numpy array.") from e
    except ValueError as e:
        if len(img_rgb.shape) < 3:
            raise ValueError(
                "Input image must have at least three dimensions (height, width, channels)."
            ) from e
        elif img_rgb.shape[2] != 3:
            raise TypeError("Input image must be in RGB format.") from e
        elif bbox_height_start >= bbox_height_end or bbox_width_start >= bbox_width_end:
            raise ValueError("Bounding box coordinates are invalid.") from e
        else:
            raise ValueError("Unknown error occurred.") from e


def assert_dict_keys_and_types(keys, dictionary, value_types, dictionary_name):
    for key in keys:
        assert (
            key in dictionary
        ), f"Required key '{key}' not found in {dictionary_name}."
        assert any(
            isinstance(dictionary[key], value_type) for value_type in value_types
        ), f"{dictionary_name}['{key}'] must be one of the types: {[value_type.__name__ for value_type in value_types]}"


def check_args_consistency(args_list):
    assert all(arg for arg in args_list) or all(
        not arg for arg in args_list
    ), "Arguments should either be all None/False or all not None/True."


def parse_yaml_file(yaml_file, args):
    """
    Adds arguments from a YAML file to an argument class.
    Arguments:
        yaml_file (str): Path to the YAML file containing arguments.
        args (argparse.Namespace): An argument class.
    """
    if yaml_file:
        print("Using YAML file to parse arguments, ignoring other arguments")
        with open(yaml_file, "r") as f:
            yaml_args = yaml.safe_load(f)

        # Update args with YAML arguments if present
        for key, value in yaml_args.items():
            setattr(args, key, value)

    return args


def parse_yaml_to_dict(yaml_file):
    """
    Adds arguments from a YAML file to an argument class.
    Arguments:
        yaml_file (str): Path to the YAML file containing arguments.
        args (argparse.Namespace): An argument class.
    """
    args_dict = {}

    if yaml_file:
        print("Using YAML file to parse arguments, ignoring other arguments")
        with open(yaml_file, "r") as f:
            yaml_args = yaml.safe_load(f)

        # Update args with YAML arguments if present
        for key, value in yaml_args.items():
            args_dict[key] = value

    return args_dict


def check_pip_environment(requirements_file="env/requirements-frozen.txt"):
    """
    Checks to see if the current environment matches the requirements (as determined by pip).

    Parameters:
    -----------
        requirements_file (str): The path to the requirements file to check against.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )

    installed = set((line for line in result.stdout.split("\n") if line.strip() != ""))

    with open(requirements_file) as f:
        requirements_txt = f.read()

    expected = set(
        (line for line in requirements_txt.split("\n") if line.strip() != "")
    )

    differences = installed.symmetric_difference(expected)

    return sorted(expected), sorted(installed), sorted(differences)


def custom_unravel_index(
    indices: torch.Tensor, shape: Union[int, Sequence[int], torch.Size]
) -> Tuple[torch.Tensor, ...]:
    """
    Converts a tensor of flat indices into a tuple of coordinate tensors.
    """
    # Validate input tensor type
    if (
        not indices.dtype.is_floating_point
        and indices.dtype != torch.bool
        and not indices.is_complex()
    ):
        pass
    else:
        raise ValueError("expected 'indices' to be an integer tensor")

    # Ensure shape is in correct format
    if isinstance(shape, int):
        shape = torch.Size([shape])
    elif isinstance(shape, Sequence):
        for dim in shape:
            if not isinstance(dim, int):
                raise ValueError("expected 'shape' sequence to contain only integers")
        shape = torch.Size(shape)
    else:
        raise ValueError("expected 'shape' to be an integer or sequence of integers")

    # Check for non-negative dimensions
    if any(dim < 0 for dim in shape):
        raise ValueError("'shape' cannot have negative values")

    # Calculate coefficients for unraveling
    coefs = list(
        reversed(
            list(
                itertools.accumulate(
                    reversed(shape[1:] + torch.Size([1])), func=operator.mul
                )
            )
        )
    )

    # Return from original
    # indices.unsqueeze(-1).floor_divide(
    #     torch.tensor(coefs, device=indices.device, dtype=torch.int64)
    # ) % torch.tensor(shape, device=indices.device, dtype=torch.int64)

    indices = indices.unsqueeze(-1)
    coefs_tensor = torch.tensor(coefs, device=indices.device, dtype=torch.int64)
    shape_tensor = torch.tensor(shape, device=indices.device, dtype=torch.int64)

    unravelled_indices = (
        torch.div(indices, coefs_tensor, rounding_mode="floor") % shape_tensor
    )

    return tuple(unravelled_indices.unbind(-1))


def init_or_update(metrics_dict, key, addition, regularization=False):
    if key not in metrics_dict or metrics_dict[key] is None:
        metrics_dict[key] = 0

    metrics_dict[key] += addition


def report_memory_status(phase=""):
    gc.collect()  # Garbage collect to free unreferenced memory
    allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert bytes to GB
    cached = torch.cuda.memory_reserved() / (1024**3)  # Convert bytes to GB
    print(f"{phase} Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")


def get_learning_rates(optimizer, model, detailed=False):
    # WARNING: this function assumes all parameters inside
    #          a complete module has the same LR

    # param_to_name will be a dict of param_id: param_name
    # e.g. {13987308184016: 'backbone.embedded_model.features.0.weight}
    param_to_name = {}
    info_lst = model.named_modules() if detailed else model.named_children()
    for name, module in info_lst:
        for param_name, param in module.named_parameters(recurse=True):
            param_full_name = f"{name}|{param_name}" if name else param_name
            param_to_name[id(param)] = param_full_name

    # if detailed, names would be backbone.fc1|weight
    # else, names would be backbone|fc1.weight
    # and they will be processed later on

    # lr dict will be a dict of layer_name: lr
    # e.g. {'add_on_layers.backbone.blabla': 0.01}
    lr_dict = {}  # module-param-lr
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if id(param) in param_to_name.keys():
                key = f"lr.{param_to_name[id(param)].split('|')[0]}"
            else:
                key = f"Unnamed Parameter {id(param)}"
            lr_dict[key] = param_group["lr"]

    return lr_dict


def predicated_extend(predicate, list1, list2):
    if predicate:
        list1.extend(list2)
    return list1
