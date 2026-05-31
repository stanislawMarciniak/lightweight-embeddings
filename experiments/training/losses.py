from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    vx = torch.sqrt((x ** 2).mean() + 1e-8)
    vy = torch.sqrt((y ** 2).mean() + 1e-8)
    corr = (x * y).mean() / (vx * vy + 1e-8)
    return corr


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - pearson_corr(pred, target)


def soft_rank(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:

    x = x.view(-1)
    n = x.size(0)
    x_diff = x.view(n, 1) - x.view(1, n)
    P_hat = torch.sigmoid(-x_diff / tau)
    ranks = P_hat.sum(dim=-1) + 0.5  
    return ranks


def soft_spearman_loss(
    pred: torch.Tensor, target: torch.Tensor, tau: float = 1.0, eps: float = 1e-8
) -> torch.Tensor:

    pred_rank = soft_rank(pred, tau)
    target_rank = soft_rank(target, tau)

    vx = pred_rank - pred_rank.mean()
    vy = target_rank - target_rank.mean()
    corr = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2)) + eps
    )
    corr = torch.clamp(corr, -1.0, 1.0)
    return 1.0 - corr


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Soft margin-based contrastive loss on cosine distance.
    Labels are continuous [0, 1]: high label = similar pair (pull together),
    low label = dissimilar pair (push apart). No binarization — full gradient
    signal is preserved across the label range.
    """
    cos_sim = F.cosine_similarity(z1, z2)
    dist = 1.0 - cos_sim

    pos_loss = labels * (dist ** 2)
    neg_loss = (1.0 - labels) * (F.relu(margin - dist) ** 2)
    return (pos_loss + neg_loss).mean()


class HybridLoss(nn.Module):
    def __init__(
        self,
        w_pearson: float = 0.2,
        w_spearman: float = 0.2,
        w_contrastive: float = 0.6,
        tau_spearman: float = 1,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        total = w_pearson + w_spearman + w_contrastive
        self.w_pearson = w_pearson / total
        self.w_spearman = w_spearman / total
        self.w_contrastive = w_contrastive / total
        self.tau_spearman = tau_spearman
        self.margin = margin

    def forward(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        lp = pearson_loss(score, target)
        ls = soft_spearman_loss(score, target, tau=self.tau_spearman)
        lc = contrastive_loss(z1, z2, target, margin=self.margin)

        total = self.w_pearson * lp + self.w_spearman * ls + self.w_contrastive * lc
        return total, {
            "pearson": lp.item(),
            "spearman": ls.item(),
            "contrastive": lc.item(),
        }