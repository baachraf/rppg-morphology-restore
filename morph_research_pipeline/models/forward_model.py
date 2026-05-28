"""
models/forward_model.py — Differentiable PPG → rPPG Forward Model
==================================================================
Learns the camera measurement process: given a true PPG waveform,
predicts what rPPG signal the CHROM algorithm would extract.

Used by DPS (A10) as the differentiable likelihood model:
  P(rPPG_obs | PPG) ∝ exp(-||PPGToRPPG(PPG) - rPPG_obs||²_pearson)

Architecture: 4-layer 1D residual Conv network (~50K params)
  Input:  (B, 1, 256) GT PPG cycle in [0, 1]
  Output: (B, 1, 256) predicted CHROM rPPG cycle (z-scored space)

Trained with Pearson correlation loss on aligned (gt_cycles, rppg_chrom_cycles)
pairs from the training split.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv1d(ch, ch, 5, 1, 2),
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv1d(ch, ch, 5, 1, 2),
        )

    def forward(self, x):
        return x + self.net(x)


class PPGToRPPG(nn.Module):
    """
    Differentiable PPG → CHROM rPPG forward model.
    ~50K parameters.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.in_conv  = nn.Conv1d(1, hidden, 7, 1, 3)
        self.res1     = ResBlock(hidden)
        self.res2     = ResBlock(hidden)
        self.res3     = ResBlock(hidden)
        self.res4     = ResBlock(hidden)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(),
            nn.Conv1d(hidden, 1, 7, 1, 3),
        )

    def forward(self, ppg):
        """ppg: (B, 1, 256) in [0, 1] → rppg_est: (B, 1, 256)"""
        h = self.in_conv(ppg)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        return self.out_conv(h)
