"""
This module defines the paths for data and outputs using a standardized,
deploy-ready project structure, and stores all global project-wide configuration constants.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Path Configuration (Standardized Deployment Structure)
# ---------------------------------------------------------------------------

# Since this file lives in project_root/src/, we get the project root directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Standardized folder locations
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def get_dataset_paths():
    """
    Returns the dataset paths for particles and voids for both train and val sets.
    
    Structure expected:
        project_root/
        ├── data/
        │   ├── voxel_particles_train/
        │   ├── voxel_particles_val/
        │   ├── voxel_void_train/
        │   └── voxel_void_val/
    """
    root_particles = os.path.join(DATA_DIR, "voxel_particles_")
    root_void = os.path.join(DATA_DIR, "voxel_void_")

    return (
        root_particles + "train" + os.sep,
        root_particles + "val" + os.sep,
        root_void + "train" + os.sep,
        root_void + "val" + os.sep,
    )


def get_output_paths():
    """
    Returns the output path for models and logs.
    
    Structure expected:
        project_root/
        └── output/
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR + os.sep


# ---------------------------------------------------------------------------
# Project-wide constants
# ---------------------------------------------------------------------------
PREFIX = "merged_labeled_zone"
CUBESIZE = 2000 / 16  # Physical size of a voxel cube in Mpc/h

NUM_HEADS = 4  # Detection heads (scales: 2x2x2, 4x4x4, 8x8x8)
NUM_ANCHORS = 5  # Anchors per head
SUBSETS_PER_ANCHOR = 3  # Consecutive radius subsets merged per anchor

ANCHOR_MULTIPLIERS = (
    2 ** (np.arange(-NUM_ANCHORS // 2 + 1, NUM_ANCHORS // 2 + 1) / NUM_ANCHORS)[::-1]
)
N_BINS = NUM_ANCHORS * NUM_HEADS
SCALE_ADJ = 2

