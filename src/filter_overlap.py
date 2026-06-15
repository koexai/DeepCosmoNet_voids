"""
This filter mimics the pylians filter where smaller voids overlapped
with bigger ones are deleted
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def filter_smaller_touching_spheres(touches):
    """
    Given a boolean matrix where touches[i, j] is True if sphere i and j overlap.
    Assumes spheres are ordered from largest to smallest.
    """
    num_spheres = touches.shape[0]
    keep = torch.ones(num_spheres, dtype=torch.bool)

    for i in range(num_spheres):
        # Only check if the current sphere hasn't been eliminated yet
        if keep[i].item():
            # Find all smaller spheres (indices > i) that touch this sphere
            overlapping_smaller = touches[i, i + 1 :]
            # Mark those specific smaller spheres for deletion
            keep[i + 1 :][overlapping_smaller] = False

    return keep


def filter_touching_spheres(spheres):
    """Filters out smaller spheres that touch larger ones.
    Assumes 'spheres' is a DataFrame with columns 'x', 'y',
    'z' (optional) and 'radius', sorted by radius in descending order."""
    if len(spheres) > 1:
        # 1. Prepare coordinates (handles x,y or x,y,z)
        cols = [c for c in spheres.columns if c in ["x", "y", "z"]]
        xyz = torch.as_tensor(
            spheres[cols].values.astype(np.float32), dtype=torch.float32
        )

        # 2. Calculate distance matrix
        dists = torch.cdist(xyz, xyz, p=2)

        # 3. Calculate sum of radii matrix
        radii = torch.as_tensor(
            spheres["radius"].values.astype(np.float32), dtype=torch.float32
        )
        sum_r = radii.view(-1, 1) + radii.view(1, -1)

        # 4. Determine intersections
        touches = (dists - sum_r) <= 0

        # 5. Iteratively filter
        keep_mask = filter_smaller_touching_spheres(touches)

        return spheres[keep_mask.numpy()]
    else:
        return spheres


def run_test_and_plot(n_points=1000):
    """Generates random spheres, applies the filter, and plots before/after."""
    # Generate random 2D data
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x": np.random.uniform(0, 100, n_points),
            "y": np.random.uniform(0, 100, n_points),
            "r": np.exp(np.random.uniform(0.5, 2.5, n_points)),
        }
    )

    # Ensure it's sorted by radius (Big to Small)
    df = df.sort_values(by="r", ascending=False).reset_index(drop=True)

    # Apply filter
    df_filtered = filter_touching_spheres(df)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, data, title in zip(
        [ax1, ax2], [df, df_filtered], ["Before Filter", "After Filter"]
    ):
        ax.set_aspect("equal")
        ax.set_title(f"{title} (Count: {len(data)})")
        for _, row in data.iterrows():
            circle = plt.Circle(
                (row["x"], row["y"]), row["r"], color="blue", alpha=0.3, ec="black"
            )
            ax.add_patch(circle)
        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 110)

    fig.savefig("touching_spheres_filter_test.png")
    plt.show()


# Run the test
if __name__ == "__main__":
    run_test_and_plot(1000)
