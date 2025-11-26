# model_spectraformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    1D Conv block: Conv(k=3, stride=2) -> BN -> ReLU -> Conv(k=1) -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.conv1 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        return x


class TransformerBlock(nn.Module):
    """
    Simple transformer encoder block with 2-head self-attention and MLP,
    applied to the sequence dimension (wavelength positions).
    """
    def __init__(self, d_model, n_heads=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=n_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L) -> treat L as sequence length, C as embedding dim
        x = x.transpose(1, 2)  # (B, L, C)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        x = x.transpose(1, 2)  # back to (B, C, L)
        return x


class Spectraformer(nn.Module):
    """
    Spectraformer for 1D NIR spectra classification.
    Input: (B, 1, 331) by default.
    """
    def __init__(self, input_length=331, num_classes=24,
                 base_channels=16, dropout=0.3):
        super().__init__()

        # First conv block (16 channels)
        self.block1 = ConvBlock(1, base_channels)

        # Transformer after first block (on C=16 channels, 2 heads)
        self.transformer = TransformerBlock(d_model=base_channels,
                                            n_heads=2,
                                            dim_ff=4 * base_channels,
                                            dropout=dropout)

        # Remaining conv blocks, doubling channels each time
        self.block2 = ConvBlock(base_channels, base_channels * 2)    # 16 -> 32
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4)  # 32 -> 64
        self.block4 = ConvBlock(base_channels * 4, base_channels * 8)  # 64 -> 128

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        in_fc = base_channels * 8
        self.fc1 = nn.Linear(in_fc, in_fc // 2)
        self.fc2 = nn.Linear(in_fc // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 1, L)
        x = self.block1(x)
        x = self.transformer(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x), inplace=True))
        x = self.fc2(x)
        return x
    
