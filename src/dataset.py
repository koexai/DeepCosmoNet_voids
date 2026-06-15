"""
This module aims to format the data at the input and output of the network
with multi-anchor support and sliding window on radius subsets.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from src.dcn_config import (
    get_dataset_paths,
    PREFIX,
    CUBESIZE,
    NUM_HEADS,
    NUM_ANCHORS,
    SUBSETS_PER_ANCHOR,
    ANCHOR_MULTIPLIERS,
    N_BINS,
    SCALE_ADJ,
)
from src.augmentation import random_flip, random_permute
from src.features import prepare_void_detection_input
from src.full_grid_tensor import voids_to_target_full_grid, BoxClusterer, group_means


def sort_voids_by_radius(voids):
    """
    Sorts voids from a dataframe into subsets based on radius bins.
    Uses a fine-grained logarithmic scale for sliding window anchor assignment.

    Args:
        pandas Dataframe: voids with columns 'x', 'y', 'z', 'radius'

    Returns:
        list of Dataframes: voids subsets sorted by radius in descending order
    """
    n_subsets = N_BINS + SUBSETS_PER_ANCHOR - 1
    limits = (
        1.95
        * (2 ** (6 - NUM_HEADS) * 2 ** (np.arange(-1, n_subsets) / NUM_ANCHORS))[::-1]
    )

    subsets = []
    for i in range(len(limits) - 1):
        min_r, max_r = limits[i + 1], limits[i]
        subset = voids[(voids["radius"] >= min_r) & (voids["radius"] < max_r)]
        subsets.append(subset)

    return subsets


def target_to_voids(tensor, anchor, o_x, o_y, o_z, thresh=0.0):
    """
    Opposite function of voids_to_targets.
    Args:
        tensor: Tensor of shape (B, 5, grid_size, grid_size, grid_size)
        integers o_x, o_y, o_z: the offset of the cube in the universe
        thresh: 0.5 if after sigmoid 0.0 if before sigmoid
    Returns:
        pandas Dataframe: dataframe of voids 'x', 'y', 'z', 'radius'
    """
    # converts a list of lists to tensor if needed
    if isinstance(tensor, list):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    # convert numpy array to tensor if needed
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).float()
    # make it work for batches or a single target
    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    # get voids above the threshold
    mask = tensor[:, 0, :, :, :] > thresh
    ids = torch.where(mask)

    anchor_mult = ANCHOR_MULTIPLIERS[anchor]
    if len(mask) == 0:
        # no object found
        return pd.DataFrame(
            {
                "x": np.array([]),
                "y": np.array([]),
                "z": np.array([]),
                "radius": np.array([]),
            }
        )

    # get grid ids of voids
    x_ids = ids[1].numpy()
    y_ids = ids[2].numpy()
    z_ids = ids[3].numpy()

    # get offsets for detected voids
    x_off = tensor[:, 1, x_ids, y_ids, z_ids].numpy()
    y_off = tensor[:, 2, x_ids, y_ids, z_ids].numpy()
    z_off = tensor[:, 3, x_ids, y_ids, z_ids].numpy()
    r_off = tensor[:, 4, x_ids, y_ids, z_ids].numpy()

    # get scale
    grid_size = tensor.shape[-1]
    scale = grid_size / CUBESIZE

    # add grid offset and voxel offset
    x = (x_ids + x_off * SCALE_ADJ + 0.5 + o_x * grid_size) / scale
    y = (y_ids + y_off * SCALE_ADJ + 0.5 + o_y * grid_size) / scale
    z = (z_ids + z_off * SCALE_ADJ + 0.5 + o_z * grid_size) / scale

    # invert radius log scaled
    radius = (np.power(2, r_off - 0.5) / scale) * SCALE_ADJ * anchor_mult

    return pd.DataFrame(
        {
            "x": np.ravel(x),
            "y": np.ravel(y),
            "z": np.ravel(z),
            "radius": np.ravel(radius),
        }
    )


class VoidDataset(Dataset):
    """
    Overrides the PyTorch DataSet Class specific for our use case
    with multi-anchor support.
    """

    def __init__(self, particles_dir, voids_dir, augment):
        self.particles_dir = particles_dir
        self.voids_dir = voids_dir
        self.resolution = 128
        self.cached_items = {}
        self.augment = augment

        self.box_fuser = BoxClusterer()

        # find particle files
        pattern = os.path.join(particles_dir, PREFIX + "_*.parquet")
        particle_files = sorted(glob.glob(pattern))

        self.file_indices = []
        for file_path in particle_files:
            filename = os.path.basename(file_path)
            # match = re.search(PREFIX+r'_(\d+_\d+_\d+)\.parquet', filename)
            match = re.search(
                PREFIX + r"_([0-9a-f]+_[0-9a-f]+_[0-9a-f]+)\.parquet", filename
            )
            if match:
                self.file_indices.append(match.group(1))

        print("Loaded", len(self.file_indices), "files")
        if len(self.file_indices) == 0:
            print(os.listdir(particles_dir))
            print(os.listdir(voids_dir))
            exit()

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the specified index.

        Multi-anchor version: generates 15-channel targets for each of 3 heads.
        Each head has 3 anchors, each anchor uses 3 consecutive radius subsets.

        Parameters:
        - index: int, index of the sample to retrieve

        Returns:
        - sample: a dataset sample with input and output tensors ready for the model
            - input: (5, 128, 128, 128)
            - targets: list of 3 tensors, shapes (15, 2, 2, 2), (15, 4, 4, 4), (15, 8, 8, 8)
        """
        if index in self.cached_items:
            return self.cached_items[index]

        file_index = self.file_indices[index]
        particle_file = os.path.join(
            self.particles_dir, f"{PREFIX}_{file_index}.parquet"
        )
        void_file = os.path.join(self.voids_dir, f"voids_{file_index}.parquet")

        try:
            particles_df = pd.read_parquet(particle_file, engine="fastparquet")
        except FileNotFoundError as fnf:
            raise FileNotFoundError(
                f"particle files not found: {particle_file}"
            ) from fnf

        try:
            voids_df = pd.read_parquet(void_file, engine="fastparquet")
        except FileNotFoundError as fnf:
            raise FileNotFoundError(f"voids file not found: {void_file}") from fnf

        # o_x, o_y, o_z = np.int32(file_index.split("_"))
        o_x, o_y, o_z = [int(x, 16) for x in file_index.split("_")]

        if self.augment:
            voids_df, particles_df, o_x, o_y, o_z = random_permute(
                voids_df, particles_df, o_x, o_y, o_z
            )
            voids_df, particles_df, o_x, o_y, o_z = random_flip(
                voids_df, particles_df, o_x, o_y, o_z
            )

        # Create 11 fine-grained radius subsets
        voids_subsets_all = sort_voids_by_radius(voids_df)  # List of 11 dataframes

        yolo_targets = []

        # Process 3 heads (scales: 2x2x2, 4x4x4, 8x8x8)
        for head in range(NUM_HEADS):
            grid_size = 2 ** (head + 1)  # 2, 4, 8
            scale = grid_size / CUBESIZE

            # Storage for 3 anchors × 5 channels = 15 channels
            head_targets = []

            # Process 3 anchors for this head
            for anchor in range(NUM_ANCHORS):
                # Calculate which 3 consecutive subsets this anchor uses
                # Sliding window: each anchor shifts by 1 subset
                base_idx = head * NUM_ANCHORS + anchor
                subset_indices = [base_idx, base_idx + 1, base_idx + 2]
                # print(subset_indices)
                # print(len(voids_subsets_all))

                # Merge the 3 consecutive subsets for this anchor
                anchor_voids = pd.concat(
                    [voids_subsets_all[i] for i in subset_indices], ignore_index=True
                )

                # Apply WBF (Weighted Box Fusion) if there are multiple voids
                if len(anchor_voids) > 1:
                    coords = torch.as_tensor(
                        anchor_voids[["x", "y", "z"]].values, dtype=torch.float32
                    )
                    distances = torch.cdist(coords, coords, p=2)
                    labels = self.box_fuser.cluster(distances * scale)
                    fused_voids = group_means(anchor_voids, labels)

                    # print(f"WBF head {head} anchor {anchor} (grid={grid_size}, subsets={subset_indices}): "
                    # f"{len(anchor_voids)} -> {len(fused_voids)} voids")

                    anchor_voids = fused_voids

                # Generate 5-channel target for this anchor
                anchor_target = torch.as_tensor(
                    voids_to_target_full_grid(
                        anchor_voids, anchor, grid_size, o_x, o_y, o_z
                    )
                )

                head_targets.append(anchor_target)

            # Concatenate 3 anchors: (5, H, W, D) × 3 → (15, H, W, D)
            head_target_15ch = torch.cat(head_targets, dim=0)
            yolo_targets.append(head_target_15ch)

        # Prepare input density
        density, _ = np.histogramdd(
            particles_df[["x", "y", "z"]].values, bins=self.resolution
        )

        input_tensor = prepare_void_detection_input(density)

        item = {
            "input": input_tensor,
            "targets": yolo_targets,  # List of tensors: one for each head
            "ox": o_x,
            "oy": o_y,
            "oz": o_z,
        }

        # self.cached_items[index] = item
        return item

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: dataset length
        """
        return len(self.file_indices)


def void_data_loader(batch_size=1):
    """
    Creates PyTorch DataLoader objects for training and validation datasets.

    Parameters:
    - batch_size: int, size of each batch for data loading (default: 16)

    Returns:
    - train_loader, val_loader: two PyTorch DataLoader objects for train and validation
    """
    train_part_path, val_part_path, train_void_path, val_void_path = get_dataset_paths()
    train = VoidDataset(
        particles_dir=train_part_path, voids_dir=train_void_path, augment=True
    )
    validation = VoidDataset(
        particles_dir=val_part_path, voids_dir=val_void_path, augment=False
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        validation,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, valid_loader
