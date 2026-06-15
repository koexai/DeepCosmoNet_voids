"""
Cosmic Void Detection Network - Modular YOLO v1 3D with Multi-Scale Heads
This implementation defines a flexible 3D detector inspired by YOLO v1,
adapted for cosmic void detection in 3D cosmological data.
The architecture includes a backbone with configurable depth,
a feature pyramid with fusion, and multiple detection heads.
"""

import torch
from torch import nn
from src.dcn_config import (
    ANCHOR_MULTIPLIERS as _ANCHOR_MULTIPLIERS,
    NUM_ANCHORS as _NUM_ANCHORS,
    NUM_HEADS,
)


class Config3D:
    """Configuration class for the 3D YOLO-based void detection network."""
    # Scale grids for num heads
    SCALES = [2, 4, 8, 16][: NUM_HEADS + 1]

    # Anchor multipliers for radius prediction (log-scale), from config_env
    ANCHOR_MULTIPLIERS = _ANCHOR_MULTIPLIERS
    NUM_ANCHORS = _NUM_ANCHORS

    CHANNELS_PER_ANCHOR = 5
    TARGET_CHANNELS = CHANNELS_PER_ANCHOR * NUM_ANCHORS

    INPUT_SIZE = (128, 128, 128)
    INPUT_CHANNELS = 5


config3d = Config3D()


class ConvBlock3D(nn.Module):
    """
    Reusable 3D convolutional block.

    Parameters:
    - in_channels: int, number of input channels
    - out_channels: int, number of output channels
    - kernel_size: int, convolution kernel size (default: 2)
    - stride: int, convolution stride (default: 2)
    - padding: int, convolution padding (default: 0)
    - dropout_rate: float, dropout probability (default: 0.0)

    Returns:
    - tensor: processed feature tensor
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        padding=0,
        dropout_rate=0.0,
    ):
        """ConvBlock3D: 3D convolution
        followed by BN, activation, and dropout."""
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        """Forward pass for the convolutional block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class UpsampleBlock3D(nn.Module):
    """
    Reusable 3D upsampling block.

    Parameters:
    - in_channels: int, number of input channels
    - out_channels: int, number of output channels
    - dropout_rate: float, dropout probability (default: 0.0)

    Returns:
    - tensor: upsampled feature tensor
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        """UpsampleBlock3D: 3D transposed convolution for upsampling,
        followed by BN, activation, and dropout."""
        super().__init__()

        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        """Forward pass for the upsampling block."""
        x = self.upsample(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FusionBlock3D(nn.Module):
    """
    Reusable feature fusion block.

    Parameters:
    - in_channels: int, number of input channels
    - out_channels: int, number of output channels

    Returns:
    - tensor: fused feature tensor
    """

    def __init__(self, in_channels, out_channels):
        """FusionBlock3D: 1x1x1 convolution
        to fuse features from different levels,"""
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        """Forward pass for the fusion block."""
        x = self.conv(x)
        x = self.activation(x)
        return x


class SuperSeparableBlock(nn.Module):
    """
    A residual hidden block to reduce parameters while maintaining expressiveness:
    Conv3d -> BN -> LeakyReLU -> Dropout, with skip connection.
    """

    def __init__(self, channels, dropout_rate=0.0):
        """SuperSeparableBlock initialization """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding="same",
                groups=channels,
            ),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 1),
                stride=1,
                padding="same",
                groups=channels,
            ),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 1, 3),
                stride=1,
                padding="same",
                groups=channels,
            ),
            nn.Conv3d(
                channels, channels, kernel_size=(1, 1, 1), stride=1, padding="same"
            ),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        """Forward pass for the SuperSeparableBlock with a skip connection."""
        return x + self.block(x)  # skip connection


class Detection3DHead(nn.Module):
    """
    Flexible 3D detection head with multiple anchors.

    Parameters:
    - in_channels: int, number of input channels
    - grid_size: int, size of the output grid
    - dropout_rate: float, dropout probability (default: 0.0)
    - num_hidden_blocks: int, number of hidden conv blocks to insert

    Returns:
    - tensor: shape (5*NUM_ANCHORS, grid_size, grid_size, grid_size)
    For each anchor:
    [anchor*5 + 0] = objectness (void presence probability)
    [anchor*5 + 1] = x_offset
    [anchor*5 + 2] = y_offset
    [anchor*5 + 3] = z_offset
    [anchor*5 + 4] = radius_offset (log-scaled, relative to anchor multiplier)
    """

    def __init__(
        self,
        in_channels,
        grid_size,
        dropout_rate=0.0,
        num_hidden_blocks=1,
        initial_obj_prior=0.0,
    ):
        """Initializes the Detection3DHead with a learnable objectness prior."""
        super().__init__()

        self.grid_size = grid_size
        self.output_channels = config3d.TARGET_CHANNELS  # = 15
        grp = 2  # self.output_channels

        inner_channels = min(512, in_channels // 2)  # Adapt to input channels

        # First projection conv (reduce channels)
        layers = [
            nn.Conv3d(
                in_channels, inner_channels * grp, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm3d(inner_channels * grp),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout3d(p=dropout_rate),
        ]

        # Add residual hidden blocks
        for _ in range(num_hidden_blocks):
            layers.append(
                SuperSeparableBlock(inner_channels * grp, dropout_rate=dropout_rate)
            )

        # Output layer
        layers += [
            nn.Conv3d(
                inner_channels * grp,
                self.output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
            ),
        ]

        self.layers = nn.Sequential(*layers)

        # Learnable objectness prior: one scalar per anchor.
        # Initialized to a negative value so sigmoid(prior) ≈ 0.018,
        # meaning the network starts by predicting few voids.
        # The loss gradient adjusts it to match the true data distribution
        # within the first few iterations.
        self.objectness_prior = nn.Parameter(
            torch.full((config3d.NUM_ANCHORS,), initial_obj_prior)
        )

        print(
            f"Detection head {self.grid_size}x{self.grid_size}x{self.grid_size}: "
            f"learnable objectness_prior init={initial_obj_prior:.1f} "
            f"(sigmoid={torch.sigmoid(torch.tensor(initial_obj_prior)):.4f})"
        )

    def forward(self, x):
        """Forward pass for the detection head,
        adding the learnable prior to objectness channels."""
        out = self.layers(x)

        # Add the learnable prior to objectness channels only.
        # out shape: (B, NUM_ANCHORS*5, G, G, G)
        # objectness channels: 0, 5, 10, ... (every 5th channel)
        for a in range(config3d.NUM_ANCHORS):
            obj_ch = a * config3d.CHANNELS_PER_ANCHOR  # 0, 5, 10, ...
            out[:, obj_ch] = out[:, obj_ch] + self.objectness_prior[a]

        return out


class CosmicVoidDetectionVNet(nn.Module):
    """
    YOLO v1 3D with multi-scale heads for Void Detection.
    Each head has multiple anchors with their own radius multipliers
    Compatible with existing dataset.

    Parameters:
    - width_multiplier: float, multiplier for network width (default: 1.0)
    - depth_layers: list, additional layers for each depth level
    - dropout_rate: float, base dropout rate (default: 0.0)

    Input:
    - tensor: shape (batch_size, INPUT_CHANNELS, 128, 128, 128)

    Returns:
    - list: of tensors, one for each scale or head
    [
    - Scale 2x2x2: (batch, 5*N_ANCHORS, 2, 2, 2)
    - Scale 4x4x4: (batch, 5*N_ANCHORS, 4, 4, 4)
    - Scale 8x8x8: (batch, 5*N_ANCHORS, 8, 8, 8)
    [...]
    }

    Each tensor has 5*N_ANCHORS channels:
    For each anchor there is a multiplier:
    [anchor*5 + 0] = objectness (void presence probability)
    [anchor*5 + 1] = x_offset
    [anchor*5 + 2] = y_offset
    [anchor*5 + 3] = z_offset
    [anchor*5 + 4] = radius_offset (log-scaled, relative to anchor)
    """

    def __init__(
        self,
        width_multiplier=2.0,
        depth_layers=(5, 2, 2, 2, 2),
        dropout_rate=0,
        device="cuda",
    ):
        """Initializes the Cosmic Void Detection Network."""
        super().__init__()
        self.device = device

        # Calculate channels based on width_multiplier
        base_channels = [32, 64, 128, 256, 512, 1024]
        channels = [max(8, int(ch * width_multiplier)) for ch in base_channels]

        # Backbone - Encoder layers with configurable depth
        self.conv_blocks = nn.ModuleList()

        # Level 1: 4 -> channels[0] -> channels[1]
        level1_blocks = [
            ConvBlock3D(config3d.INPUT_CHANNELS, channels[0], dropout_rate=dropout_rate)
        ]
        for _ in range(depth_layers[0] - 1):
            level1_blocks.append(
                SuperSeparableBlock(channels[0], dropout_rate=dropout_rate)
            )
        level1_blocks.append(
            ConvBlock3D(channels[0], channels[1], dropout_rate=dropout_rate)
        )
        self.conv_blocks.append(nn.Sequential(*level1_blocks))

        # Levels 2-5
        for i in range(1, 5):
            level_blocks = [
                ConvBlock3D(channels[i], channels[i + 1], dropout_rate=dropout_rate)
            ]  # if i < 4 else 0.4)]
            for _ in range(depth_layers[i] - 1):
                level_blocks.append(
                    SuperSeparableBlock(channels[i + 1], dropout_rate=dropout_rate)
                )
            self.conv_blocks.append(nn.Sequential(*level_blocks))

        # Upsampling layers for FPN (using dynamic channels)
        self.upsample4 = UpsampleBlock3D(channels[5], channels[4])  # 1024 -> 512
        self.upsample3 = UpsampleBlock3D(channels[5], channels[3])  # 512 -> 256
        self.upsample2 = UpsampleBlock3D(channels[4], channels[2])  # 256 -> 128
        self.upsample1 = UpsampleBlock3D(channels[3], channels[1])  # 128 -> 64

        # Feature fusion layers (using dynamic channels)
        self.fusion4 = FusionBlock3D(channels[5], channels[5])  # 1024 -> 512
        self.fusion3 = FusionBlock3D(channels[4], channels[4])  # 512 -> 256
        self.fusion2 = FusionBlock3D(channels[3], channels[3])  # 256 -> 128
        self.fusion1 = FusionBlock3D(channels[2], channels[2])  # 128 -> 64

        # Detection heads for each scale (with dynamic number of channels)
        # Scale 2x2x2 uses channels[3] (e.g., 256)
        # Scale 4x4x4 uses channels[4] (e.g., 512)
        # Scale 8x8x8 uses channels[5] (e.g., 1024)
        # [...]
        feature_channels = [channels[5], channels[5], channels[4], channels[3]]
        self.detection_heads = nn.ModuleList(
            [
                Detection3DHead(
                    feature_channels[i],
                    config3d.SCALES[i],
                    dropout_rate=0.0,
                    num_hidden_blocks=(i + 1),
                )
                for i in range(len(config3d.SCALES))
            ]
        )

    def initialize_for_void_detection(self):
        """
        Initializes the network to optimize cosmic void detection.

        This method can be called after model instantiation
        to apply specialized initializations.
        """
        print("Initializing network for cosmic void detection...")

        # Detection heads are already initialized in their __init__
        # but we can add other initializations here if needed

        # Example: more conservative initialization of final weights
        for head in self.detection_heads:
            # Reduce variance of last layer weights for initial stability
            last_conv = None
            for module in head.layers:
                if isinstance(module, nn.Conv3d):
                    last_conv = module

            if last_conv is not None:
                # Reduce weight variance for more stable convergence
                with torch.no_grad():
                    last_conv.weight.data *= 0.1  # Reduce standard initialization

        print("Network initialization completed!")

    def forward(self, x):
        """
        Forward pass of the cosmic void detection network.

        Parameters:
        - x: tensor, input of shape (batch_size, INPUT_CHANNELS, 128, 128, 128)

        Returns:
        - list:  of tensors, one for each scale
        Each tensor has shape: (batch, 5*N_ANCHORS, grid_size, grid_size, grid_size)
        """
        # Backbone - Encoder path with configurable depth
        encoder_features = []
        current_x = x
        for conv_block in self.conv_blocks:
            current_x = conv_block(current_x)
            encoder_features.append(current_x)

        # Assign extracted features (5 levels from encoder)
        _x1, x2, x3, x4, x5 = encoder_features

        # Feature pyramid with fusion and iterative masking
        # Scale 2x2x2 (smallest, deepest features)
        feat_2x2x2 = x5
        det2 = self.detection_heads[0](feat_2x2x2)

        # Scale 4x4x4
        up4 = self.upsample4(feat_2x2x2)
        feat_4x4x4 = self.fusion4(torch.cat([up4, x4], dim=1))
        det4 = self.detection_heads[1](feat_4x4x4)

        # Scale 8x8x8
        up3 = self.upsample3(feat_4x4x4)
        feat_8x8x8 = self.fusion3(torch.cat([up3, x3], dim=1))
        det8 = self.detection_heads[2](feat_8x8x8)

        # Scale 16x16x16
        if NUM_HEADS > 3:
            up2 = self.upsample2(feat_8x8x8)
            feat_16x16x16 = self.fusion2(torch.cat([up2, x2], dim=1))
            det16 = self.detection_heads[3](feat_16x16x16)
            return [det2, det4, det8, det16]

        # Return only the 3 detections
        return [det2, det4, det8]
