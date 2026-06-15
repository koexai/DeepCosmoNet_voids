"""Test script for the full grid tensor conversion and related processing functions.
This script creates a simple dataset with a single void, runs the full grid tensor conversion,
and prints the resulting tensor summary and predicted voids after clustering."""

import torch
import pandas as pd
from full_grid_tensor import (
    voids_to_target_full_grid,
    print_tensor_summary,
    BoxClusterer,
    group_means,
)
from dataset import target_to_voids


def test_voids_to_target_functions():
    """Test function for voids_to_target_full_grid and related processing functions."""
    data = {
        "x": [62],  # world coordinate center
        "y": [62],
        "z": [2],
        "radius": [62],  # fairly large radius
    }
    subset = pd.DataFrame(data)

    grid_size = 2
    o_x, o_y, o_z = 0, 0, 0  # no offset

    print("\nRunning full-grid function with masking:\n")
    output_full = voids_to_target_full_grid(subset, 0, grid_size, o_x, o_y, o_z)
    print_tensor_summary(output_full)

    df_pred = target_to_voids(output_full, anchor=0, o_x=o_x, o_y=o_y, o_z=o_z, thresh=0.0)

    pred_coords = torch.as_tensor(df_pred[["x", "y", "z"]].values, dtype=torch.float32)
    distances = torch.cdist(pred_coords, pred_coords, p=2)

    unit_dist = grid_size / 2 * 16

    clusterer = BoxClusterer()
    labels = clusterer.cluster(distances / unit_dist)
    df_pred = group_means(df_pred, labels)

    print(df_pred)


if __name__ == "__main__":
    test_voids_to_target_functions()
