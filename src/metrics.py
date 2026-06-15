"""
This module contains the additional detection metrics
to ensure the effective match between the predicted voids and the gt ones
"""
import numpy as np
import torch

TOLERANCE = 0.5


def spheres_iou(ra, rb, d):
    """
    Compute average IoU between pairs of spheres (GPU-optimized, no boolean indexing).
    All computations are batched with torch.where.
    """

    ra = torch.as_tensor(ra)
    rb = torch.as_tensor(rb)
    d = torch.as_tensor(d)

    d = torch.relu(d - TOLERANCE)
    ra = ra + TOLERANCE
    rb = rb + TOLERANCE

    # Volumes (normalized, without 4/3*pi)
    v_a = ra**3
    v_b = rb**3

    # Case 1: no overlap (v_int = 0)
    v_int_out = torch.zeros_like(d)

    # Case 2: one inside the other
    v_int_in = torch.minimum(ra**3, rb**3)

    # Case 3: partial overlap formula
    v_int_partial = (
        (d**2 + 2 * d * (ra + rb) - 3 * (ra - rb) ** 2)
        * (ra + rb - d) ** 2
        / (16 * d)
    )

    # Combine cases using torch.where
    v_int = torch.where(
        d >= (ra + rb),  # Case 1: no overlap
        v_int_out,
        torch.where(
            d <= torch.abs(ra - rb),  # Case 2: inside
            v_int_in,
            v_int_partial,  # Case 3: partial overlap
        ),
    )

    # IoU
    v_union = v_a + v_b - v_int
    iou = v_int / v_union

    iou = torch.clamp(iou, 0, 1)

    return torch.mean(iou)


def compute_cen_rad_iou_prec_rec_f1(df_pred, df_target):
    """
    Computes:
    - average relative centroid error
    - average relative radius error
    - average IoU
    - precision
    - recall
    - f1
    Pairs with distance > target radius are discarded.
    """

    # Handling the edge cases

    n_pred = len(df_pred)
    n_target = len(df_target)

    if n_pred == 0 and n_target == 0:
        # Nothing to detect
        return float("nan"), float("nan"), float("nan"), 1.0, 1.0, 1.0

    if n_pred > 0 and n_target == 0:
        # Only false positives
        return float("nan"), float("nan"), float("nan"), 0.0, float("nan"), 0.0

    if n_pred == 0 and n_target > 0:
        # Only false negatives
        return float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 0.0

    # Convert inputs
    pred_coords = torch.as_tensor(df_pred[["x", "y", "z"]].values.astype(np.float32))
    pred_radii = torch.as_tensor(df_pred["radius"].values.astype(np.float32))

    target_coords = torch.as_tensor(
        df_target[["x", "y", "z"]].values.astype(np.float32)
    )
    target_radii = torch.as_tensor(df_target["radius"].values.astype(np.float32))

    # Find closest targets
    distances = torch.cdist(pred_coords, target_coords, p=2)
    closest_indices = torch.argmin(distances, dim=0)

    closest_target_coords = pred_coords.index_select(0, closest_indices)
    closest_target_radii = pred_radii.index_select(0, closest_indices)

    # Actual centroid distances
    actual_distances = torch.norm(target_coords - closest_target_coords, dim=1)

    # Valid pairs: distance <= target radius
    valid_mask = actual_distances <= closest_target_radii
    num_valid = valid_mask.sum()

    TP = num_valid.item()
    FP = len(df_pred) - TP
    FN = len(df_target) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else float("nan")
    )

    if num_valid == 0:
        return float("nan"), float("nan"), float("nan"), precision, recall, f1

    # Safe denominator to avoid div by 0
    safe_target_radii = torch.where(
        valid_mask, closest_target_radii, torch.ones_like(closest_target_radii)
    )

    # Relative centroid error
    relative_centroid_errors = torch.where(
        valid_mask,
        torch.relu(actual_distances - 2 * TOLERANCE) / safe_target_radii,
        torch.zeros_like(actual_distances),
    )

    # Relative radius error
    relative_radius_errors = torch.where(
        valid_mask,
        torch.relu(torch.abs(target_radii - closest_target_radii) - TOLERANCE)
        / safe_target_radii,
        torch.zeros_like(target_radii),
    )

    # Mean errors (only over valid)
    centroid_error = relative_centroid_errors.sum() / num_valid
    radius_error = relative_radius_errors.sum() / num_valid

    # IoU (reuse spheres_iou)
    iou = spheres_iou(
        closest_target_radii[valid_mask],
        target_radii[valid_mask],
        actual_distances[valid_mask],
    )

    return centroid_error.item(), radius_error.item(), iou.item(), precision, recall, f1


def compute_accuracy(predictions, targets, thresh=0):
    """
    Calculates the accuracy for the object score

    Args:
        predictions: predicted tensor (B, 5, L, L, L)
        targets: target tensor (B, 5, L, L, L)
        thresh: classification threshold

    Returns:
        dict with accuracy metrics
    """
    # make it work for batches or a single target
    if len(predictions.shape) < 5:
        predictions = predictions.unsqueeze(0)  # Add batch dimension
    if len(targets.shape) < 5:
        targets = targets.unsqueeze(0)  # Add batch dimension

    target_obj = targets[:, 0]

    pred_binary = (predictions[:, 0] > thresh).float()
    correct = (pred_binary == target_obj).float()

    # foreground accuracy
    obj_mask = target_obj > thresh
    if obj_mask.sum() > 0:
        obj_accuracy = correct[obj_mask].mean()
    else:
        obj_accuracy = torch.tensor(1.0, device=predictions.device)

    # background accuracy
    noobj_mask = target_obj <= thresh
    if noobj_mask.sum() > 0:
        noobj_accuracy = correct[noobj_mask].mean()
    else:
        noobj_accuracy = torch.tensor(1.0, device=predictions.device)

    overall_accuracy = correct.mean()

    return {
        "overall_accuracy": overall_accuracy,
        "object_accuracy": obj_accuracy,
        "no_object_accuracy": noobj_accuracy,
    }
