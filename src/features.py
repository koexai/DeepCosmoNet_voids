"""
This module contains functions to prepare multi-channel input for void detection
using Gaussian smoothing at different scales.
The main function, `prepare_void_detection_input`, takes a 3D density tensor
and applies Gaussian filters with varying sigma values
to create multiple feature channels.
Each channel is then log-transformed and standardized to optimize it
for void detection tasks."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def prepare_void_detection_input(microvoxel_data):
    """
    Prepares multi-channel input for void detection.

    Parameters:
    - microvoxel_data: ndarray, shape (128, 128, 128), 3D density tensor obtained from np.histogramdd

    Returns:
    - input_tensor: torch.Tensor, shape (4, 128, 128, 128), 4-channel tensor optimized for void detection at different scales
    """
    # smooth_combined05 = gaussian_filter(microvoxel_data, sigma=0.5)
    smooth_combined1 = gaussian_filter(microvoxel_data, sigma=1)
    smooth_combined2 = gaussian_filter(microvoxel_data, sigma=2)
    smooth_combined4 = gaussian_filter(microvoxel_data, sigma=4)
    smooth_combined8 = gaussian_filter(microvoxel_data, sigma=8)
    smooth_combined16 = gaussian_filter(microvoxel_data, sigma=16)

    features = [
        smooth_combined1,
        smooth_combined2,
        smooth_combined4,
        smooth_combined8,
        smooth_combined16,
    ]
    channels = []

    for feature in features:
        # Converti a tensor
        tensor = torch.from_numpy(np.float32(feature))

        # log1p + standardization in PyTorch (velocissimo)
        log_tensor = torch.log1p(tensor)  # log(1 + x)
        mean = log_tensor.mean()
        std = log_tensor.std()
        normalized = (log_tensor - mean) / (std)

        channels.append(normalized)

    input_tensor = torch.stack(channels, dim=0)  # (5, 128, 128, 128)

    return input_tensor
