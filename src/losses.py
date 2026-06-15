"""
This module contains the definition of the loss function
and main detection metrics for the yolo network with multi-anchor support
"""

import torch
from torch import nn
import torch.nn.functional as func

# Head weight exponential multiplier
# used in focal loss calculations
hwem = 4


class YOLO3DLoss(nn.Module):
    """
    This class handles the hyperparameter of a Yolo-like loss function
    with multi-anchor support (3 anchors per detection head)
    """

    def __init__(
        self,
        objectness_weight=1,
        coordinate_weight=1,
        radius_weight=1,
        focal_alpha=4,
        focal_gamma=2,
        num_anchors=3,
    ):
        """
        Loss function per YOLO 3D with multi-anchor support

        Args:
            objectness_weight: weight for objectness loss
            coordinate_weight: weight for the center coordinate loss (x, y, z)
            radius_weight: weight for the radius loss
            focal_alpha: alpha parameter for focal loss (weight for positive class)
            focal_gamma: gamma parameter for focal loss (focusing parameter)
            num_anchors: number of anchors per head (default: 3)
        """
        super().__init__()
        
        self.objectness_weight = objectness_weight
        self.coordinate_weight = coordinate_weight
        self.radius_weight = radius_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_anchors = num_anchors
        self.channels_per_anchor = 5

        self.uncertainty_loss = UncertaintyLoss(num_lambdas=3)
        self.smoothl = nn.MSELoss(reduction="none")

    def focal_loss(self, pred, target):
        """
        Focal Loss for objectness detection with automatic handling of logits/probabilities.

        Implements the formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * CE(pred, target)
        where alpha_t is applied only to the positive class to encourage detection.

        Parameters:
        - pred: torch.Tensor, shape (grid, grid, grid), predictions
        - If logits: unbounded values
        - If probabilities: values in [0,1]
        - target: torch.Tensor, shape (grid, grid, grid), ground truth binary values (0 or 1)

        Returns:
        - torch.Tensor: normalized scalar focal loss value

        Note:
        - focal_alpha > 1: pushes model towards detection (more weight to positive class)
        - focal_gamma > 0: reduces weight of easy examples, focuses on hard ones
        """

        ce_loss = func.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_prob = torch.sigmoid(pred)

        # Calculate p_t (correct probability for each sample)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        # Calculate alpha_t: higher weight for positive class (detection)
        alpha_t = torch.where(target == 1, self.focal_alpha, 1.0)

        # Calculate focal weight: (1 - p_t)^gamma reduces weight of easy examples
        focal_weight = alpha_t * (1 - p_t).pow(self.focal_gamma)

        # Apply focal loss
        focal_loss = focal_weight * ce_loss

        # Separate positive and negative samples
        pos_mask = target > 0.5
        neg_mask = target <= 0.5

        # Balanced sampling: equal weight to pos and neg
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_loss = focal_loss[pos_mask].mean()
            # Hard negative mining: keep only hardest negatives
            neg_losses = focal_loss[neg_mask]
            num_hard_negs = min(
                pos_mask.sum().item() * 3, neg_mask.sum().item()
            )  # 3:1 ratio
            num_hard_negs = min(num_hard_negs, neg_losses.numel())
            hard_neg_losses, _ = torch.topk(neg_losses, num_hard_negs)
            neg_loss = hard_neg_losses.mean()

            return (pos_loss + neg_loss) / 2

        return focal_loss.mean()

    def compute_single_anchor_loss(self, pred_anchor, target_anchor):
        """
        Computes the loss for a single anchor.

        Parameters:
        - pred_anchor: torch.Tensor, shape (5, grid_size, grid_size, grid_size)
        - target_anchor: torch.Tensor, shape (5, grid_size, grid_size, grid_size)

        Returns:
        - dict: dictionary with losses for this anchor
        """
        device = pred_anchor.device

        pred_objectness = pred_anchor[0]  # (grid, grid, grid)
        pred_x = pred_anchor[1]
        pred_y = pred_anchor[2]
        pred_z = pred_anchor[3]
        pred_radius = pred_anchor[4]

        target_objectness = target_anchor[0]
        target_x = target_anchor[1]
        target_y = target_anchor[2]
        target_z = target_anchor[3]
        target_radius = target_anchor[4]

        obj_mask = target_objectness > 0.0
        num_pos = obj_mask.sum().float()

        # Calculate objectness loss with focal loss
        objectness_loss = self.focal_loss(pred_objectness, target_objectness)

        # Calculate coordinate loss only for positive cells
        if num_pos > 0:
            coord_loss_x = self.smoothl(pred_x * obj_mask, target_x * obj_mask)
            coord_loss_y = self.smoothl(pred_y * obj_mask, target_y * obj_mask)
            coord_loss_z = self.smoothl(pred_z * obj_mask, target_z * obj_mask)

            coordinate_loss = (
                coord_loss_x[obj_mask].mean()
                + coord_loss_y[obj_mask].mean()
                + coord_loss_z[obj_mask].mean()
            )
        else:
            coordinate_loss = torch.tensor(0.0, device=device)

        # Calculate radius loss only for positive cells
        if num_pos > 0:
            radius_loss_tensor = self.smoothl(
                pred_radius * obj_mask, target_radius * obj_mask
            )
            radius_loss = radius_loss_tensor[obj_mask].mean()
        else:
            radius_loss = torch.tensor(0.0, device=device)

        return {
            "objectness_loss": objectness_loss,
            "coordinate_loss": coordinate_loss,
            "radius_loss": radius_loss,
            "num_pos": num_pos,
        }

    def forward(self, pred, target):
        """
        Computes the loss for multi-anchor predictions.

        Parameters:
        - pred: torch.Tensor, shape (15, grid_size, grid_size, grid_size) for 3 anchors
        or (5, grid_size, grid_size, grid_size) for backward compatibility
        - target: torch.Tensor, same shape as pred

        Returns:
        - dict: dictionary with combined losses compatible with train.py
        """
        device = pred.device

        # Multi-anchor mode: loop over anchors
        total_objectness_loss = torch.tensor(0.0, device=device)
        total_coordinate_loss = torch.tensor(0.0, device=device)
        total_radius_loss = torch.tensor(0.0, device=device)
        total_num_pos = 0

        for anchor_idx in range(self.num_anchors):
            # Extract channels for this anchor
            start_ch = anchor_idx * self.channels_per_anchor
            end_ch = start_ch + self.channels_per_anchor

            pred_anchor = pred[start_ch:end_ch, ...]  # (5, H, W, D)
            target_anchor = target[start_ch:end_ch, ...]  # (5, H, W, D)

            # Compute loss for this anchor
            anchor_losses = self.compute_single_anchor_loss(pred_anchor, target_anchor)

            # Accumulate
            total_objectness_loss += anchor_losses["objectness_loss"]
            total_coordinate_loss += anchor_losses["coordinate_loss"]
            total_radius_loss += anchor_losses["radius_loss"]
            total_num_pos += anchor_losses["num_pos"].item()

        # Average over anchors
        objectness_loss = total_objectness_loss / self.num_anchors
        coordinate_loss = total_coordinate_loss / self.num_anchors
        radius_loss = total_radius_loss / self.num_anchors

        # Combine with uncertainty weighting
        total_loss = self.uncertainty_loss(
            torch.stack([objectness_loss, coordinate_loss, radius_loss])
        )

        return {
            "total_loss": total_loss,
            "objectness_loss": objectness_loss,  # Unweighted loss for monitoring
            "coordinate_loss": coordinate_loss,  # Unweighted loss for monitoring
            "radius_loss": radius_loss,  # Unweighted loss for monitoring
            "weighted_objectness_loss": objectness_loss,  # Weighted loss
            "weighted_coordinate_loss": coordinate_loss,  # Weighted loss
            "weighted_radius_loss": radius_loss,  # Weighted loss
            "num_positive_cells": total_num_pos,  # For monitoring
        }


class UncertaintyLoss(nn.Module):
    """
    Automatic loss weighting using learned uncertainty.

    Learns optimal weights for multiple loss components by treating
    task uncertainty as learnable parameters.

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics" (Kendall et al., 2018)
    """

    def __init__(self, num_lambdas=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_lambdas))

    def forward(self, losses):
        """
        Parameters:
        - losses: torch.Tensor of shape (num_lambdas,) containing individual losses

        Returns:
        - torch.Tensor: scalar combined loss

        Formula: L_total = Σ(1/(2sigma²) * L_i + log(sigma))
        where σ² = exp(log_var)
        """
        # Convert log-variance to precision (1/σ²)
        precision = torch.exp(-self.log_vars).to(losses.device)

        # Apply uncertainty weighting
        # precision * loss: weight inversely proportional to uncertainty
        # log_vars: regularization term (prevents σ → ∞)
        weighted_losses = precision * losses + self.log_vars.to(losses.device)
        return weighted_losses.sum()
