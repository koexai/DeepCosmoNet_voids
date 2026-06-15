"""
This file is meant to contain the code for augmentation
"""
import numpy as np
import pandas as pd


def random_permute(voids_df, particles_df, o_x, o_y, o_z):
    """
    In-place axis permutation for both dataframes and offsets by renaming columns only.

    Parameters:
    - voids_df: pd.DataFrame with columns ['x', 'y', 'z']
    - particles_df: pd.DataFrame with columns ['x', 'y', 'z']
    - o_x, o_y, o_z: float offsets for x, y, z

    Returns:
    - voids_df, particles_df, new_ox, new_oy, new_oz
    """

    # Original axis labels
    axes = ["x", "y", "z"]

    # Random permutation
    perm = np.random.permutation(3)
    permuted_axes = [axes[i] for i in perm]

    # Rename columns in place (relabeling, not moving data)
    voids_df = voids_df[permuted_axes + ["radius"]]
    particles_df = particles_df[permuted_axes]

    # Permute offsets the same way
    offsets = [o_x, o_y, o_z]
    new_offsets = [offsets[i] for i in perm]

    # Rename columns back to 'x', 'y', 'z' for consistency
    voids_df.columns = ["x", "y", "z"] + ["radius"]
    particles_df.columns = ["x", "y", "z"]

    o_x, o_y, o_z = new_offsets

    return voids_df, particles_df, o_x, o_y, o_z


def random_flip(voids_df, particles_df, o_x, o_y, o_z):
    """
    Randomly flips the sign of each axis (x, y, z) with 50% probability
    for both dataframes and offsets, and applies a +1 shift when flipping
    to account for uncentered coordinates (e.g., [0, 1] range).

    Parameters:
    - voids_df: pd.DataFrame with columns ['x', 'y', 'z']
    - particles_df: pd.DataFrame with columns ['x', 'y', 'z']
    - o_x, o_y, o_z: float offsets

    Returns:
    - voids_df, particles_df, new_o_x, new_o_y, new_o_z
    """

    flip = np.random.choice([1, -1], size=3)
    shift = (flip == -1).astype(int)  # 1 if flipped, 0 otherwise

    for i, axis in enumerate(["x", "y", "z"]):
        voids_df[axis] = voids_df[axis] * flip[i]
        particles_df[axis] = particles_df[axis] * flip[i]

    o_x = o_x * flip[0] - shift[0]
    o_y = o_y * flip[1] - shift[1]
    o_z = o_z * flip[2] - shift[2]

    return voids_df, particles_df, o_x, o_y, o_z


def test_augmentation():
    """Test the random_permute and random_flip functions with a tiny synthetic dataset."""
    # Create a tiny synthetic dataset
    particles_df = pd.DataFrame({"x": [0.1], "y": [0.2], "z": [0.3]})
    voids_df = pd.DataFrame({"x": [0.1], "y": [0.2], "z": [0.3], "radius": [0.4]})
    ox, oy, oz = 1, 2, 3

    # Print before
    print("Before flip:")
    print("particles:", particles_df.values)
    print("voids:", voids_df.values)
    print("offsets:", (ox, oy, oz))

    # Apply flip
    voids_df, particles_df, ox, oy, oz = random_permute(
        voids_df, particles_df, ox, oy, oz
    )
    voids_df, particles_df, ox, oy, oz = random_flip(voids_df, particles_df, ox, oy, oz)

    # Print after
    print("\nAfter flip:")
    print("particles:", particles_df.values)
    print("voids:", voids_df.values)
    print("offsets:", (ox, oy, oz))

    # Manual check (or assert logic can be added here)


if __name__ == "__main__":
    test_augmentation()
