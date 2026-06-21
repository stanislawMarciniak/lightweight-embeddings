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


def cosent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.05,
) -> torch.Tensor:
    """
    CoSENT loss (https://kexue.fm/archives/8847).

    Optimizes the *ordering* of cosine similarities to match the label order:
    for every pair (i, j) with label_i > label_j it enforces cos_i > cos_j via a
    smooth logsumexp ranking objective. This is the state-of-the-art training
    objective for STS regression — it directly targets Spearman/Pearson and adds
    zero inference cost (operates on already-computed embeddings).
    """
    cos = F.cosine_similarity(z1, z2) / tau  # (B,)
    # diff[i, j] = cos_j - cos_i  (we want this < 0 when label_i > label_j)
    diff = cos.unsqueeze(0) - cos.unsqueeze(1)
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # [i, j] = y_i - y_j
    valid = (label_diff > 0).float()  # only pairs where y_i > y_j contribute
    diff = diff - (1.0 - valid) * 1e12
    diff = diff.view(-1)
    zero = torch.zeros(1, device=diff.device, dtype=diff.dtype)
    return torch.logsumexp(torch.cat([zero, diff], dim=0), dim=0)


class HybridLoss(nn.Module):
    def __init__(
        self,
        w_pearson: float = 0.2,
        w_spearman: float = 0.2,
        w_contrastive: float = 0.3,
        w_cosent: float = 0.3,
        tau_spearman: float = 1,
        cosent_tau: float = 0.05,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        total = w_pearson + w_spearman + w_contrastive + w_cosent
        total = total if total > 0 else 1.0
        self.w_pearson = w_pearson / total
        self.w_spearman = w_spearman / total
        self.w_contrastive = w_contrastive / total
        self.w_cosent = w_cosent / total
        self.tau_spearman = tau_spearman
        self.cosent_tau = cosent_tau
        self.margin = margin

    def forward(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        total = score.new_zeros(())
        parts: dict = {}

        if self.w_pearson > 0:
            lp = pearson_loss(score, target)
            total = total + self.w_pearson * lp
            parts["pearson"] = lp.item()
        if self.w_spearman > 0:
            ls = soft_spearman_loss(score, target, tau=self.tau_spearman)
            total = total + self.w_spearman * ls
            parts["spearman"] = ls.item()
        if self.w_contrastive > 0:
            lc = contrastive_loss(z1, z2, target, margin=self.margin)
            total = total + self.w_contrastive * lc
            parts["contrastive"] = lc.item()
        if self.w_cosent > 0:
            lco = cosent_loss(z1, z2, target, tau=self.cosent_tau)
            total = total + self.w_cosent * lco
            parts["cosent"] = lco.item()

        return total, parts