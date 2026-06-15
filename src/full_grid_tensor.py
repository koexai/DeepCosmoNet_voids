"""
This module contains the main function to convert void proposals
into a full 3D grid tensor, and a reusable BoxClusterer class
for clustering boxes based on IoU."""

import torch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from src.dcn_config import CUBESIZE, ANCHOR_MULTIPLIERS, SCALE_ADJ


def voids_to_target_full_grid(subset, anchor, grid_size, o_x, o_y, o_z):
    """
    Assign every grid cell to the closest void using KDTree,
    and mask out cells that are outside the void radius.

    Args:
        subset: pandas DataFrame with columns ['x', 'y', 'z', 'radius']
        anchor: int, anchor index (0, 1, or 2) for radius scaling
        grid_size: int, size of the 3D output grid
        o_x, o_y, o_z: int offsets in voxel space

    Returns:
        tensor: shape (5, grid_size, grid_size, grid_size)
    """
    tensor = np.zeros((5, grid_size, grid_size, grid_size), dtype=np.float32)

    if len(subset) == 0:
        return tensor

    # multipliers
    anchor_mult = ANCHOR_MULTIPLIERS[anchor]

    # Scale world units into local voxel grid (correct base scale)
    scale = grid_size / CUBESIZE

    x = subset["x"].values * scale - o_x * grid_size
    y = subset["y"].values * scale - o_y * grid_size
    z = subset["z"].values * scale - o_z * grid_size
    # real radius in grid units (for mask and KDTree)
    r = subset["radius"].values * scale

    voids_pos = np.stack([x, y, z], axis=1)  # shape (N, 3)
    voids_r = r  # real radius in grid units (for the mask)

    # offset logaritmico compresso con SCALE_ADJ
    # Inverso di decode: radius = (2^(r_off-0.5) / scale) * SCALE_ADJ * anchor_mult
    voids_r_log = np.log2(r / (SCALE_ADJ * anchor_mult)) + 0.5

    # Build KDTree for efficient NN search
    tree = cKDTree(voids_pos)

    # Build the full grid of voxel centers
    grid_coords = np.stack(
        np.meshgrid(
            np.arange(grid_size),
            np.arange(grid_size),
            np.arange(grid_size),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    grid_world = (grid_coords + 0.5).astype(np.float32)  # (V, 3)

    # Query the nearest void for each grid cell
    dists, indices = tree.query(grid_world, k=1)

    matched_voids = voids_pos[indices]  # (V, 3)
    matched_radius = voids_r[indices]  # (V,)
    matched_radius_log = voids_r_log[indices]  # (V,)

    # Mask: keep cells within void radius
    mask = dists <= matched_radius * np.sqrt(2)  # (V,)

    # Compute sub-voxel offsets, compressi con SCALE_ADJ
    # Inverso di decode: x = (x_ids + x_off * SCALE_ADJ + 0.5 + o_x * G) / scale
    offsets = (matched_voids - grid_world) / SCALE_ADJ  # (V, 3)

    # Prepare tensor shape
    x_shape = (grid_size, grid_size, grid_size)

    # Fill tensor only where mask is valid
    tensor[0] = 1.0  # confidence
    tensor[1] = offsets[:, 0].reshape(x_shape)
    tensor[2] = offsets[:, 1].reshape(x_shape)
    tensor[3] = offsets[:, 2].reshape(x_shape)
    tensor[4] = matched_radius_log.reshape(x_shape)

    tensor *= mask.reshape(x_shape)

    return torch.from_numpy(tensor)


class BoxClusterer:
    """
    Reusable clusterer for grouping boxes based on precomputed IoU matrix.
    """
    def __init__(self, min_cluster_size=2, cluster_selection_epsilon=1.0):
        """
        Initialize reusable HDBSCAN clusterer.
        """
        self.clusterer = DBSCAN(
            eps=cluster_selection_epsilon,
            min_samples=min_cluster_size,
            metric="precomputed",
        )

    def cluster(self, iou_mat: torch.Tensor) -> torch.Tensor:
        """
        Cluster boxes using precomputed IoU matrix.

        Args:
            iou_mat: torch.Tensor (N,N) IoU matrix in float32

        Returns:
            labels: torch.Tensor (N,) with cluster IDs
        """
        # detach from graph + move to CPU numpy
        # dist_mat = (1.0 - iou_mat.detach().cpu().float().numpy()).astype(np.float64)
        dist_mat = iou_mat.detach().cpu().float().numpy().astype(np.float32)

        # fit & predict
        labels = self.clusterer.fit_predict(dist_mat)

        # return back as torch
        return torch.from_numpy(labels).to(iou_mat.device).long()


def group_means(
    df: pd.DataFrame, labels, exclude_outliers: bool = True
) -> pd.DataFrame:
    """
    Groups rows in df by cluster labels and averages values for clusters (label >= 0).
    Outliers (label == -1) are either dropped or appended as-is depending on exclude_outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to group.
    labels : np.ndarray
        Cluster labels from DBSCAN (same length as df).
    exclude_outliers : bool, default True
        If True, outliers (label == -1) are dropped.
        If False, outliers are appended as-is to the result.

    Returns
    -------
        pd.DataFrame with aggregated cluster means (and optionally outliers).
    """
    if len(df) != len(labels):
        raise ValueError("Length of labels must match number of rows in df")

    df = df.copy()
    df["_label"] = labels

    # Compute means for clusters (label >= 0)
    cluster_means = (
        df[df["_label"] >= 0]
        .groupby("_label")
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    cluster_means = cluster_means.drop(columns=["_label"], errors="ignore")

    if exclude_outliers:
        return cluster_means

    # Append outliers as-is
    outliers = df[df["_label"] == -1].drop(columns="_label")
    result = pd.concat([cluster_means, outliers], ignore_index=True)
    return result


def print_tensor_summary(tensor):
    """
    Nicely print the tensor content per grid cell.
    Only prints active cells (confidence > 0).
    """
    conf = tensor[0]
    x_off, y_off, z_off, r_log = tensor[1], tensor[2], tensor[3], tensor[4]

    grid_size = conf.shape[0]
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if conf[x, y, z] > 0:
                    print(
                        f"Voxel ({x},{y},{z}): conf=1, x_off={x_off[x, y, z]:.2f}, y_off={y_off[x, y, z]:.2f}, z_off={z_off[x, y, z]:.2f}, log_r={r_log[x, y, z]:.2f}"
                    )
