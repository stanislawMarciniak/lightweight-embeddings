"""On-demand fine-tuning of the custom encoder for FAQ + retrieval."""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from app.services.semantic_model import build_token_inputs
from app.training.encoder_torch import CustomEncoder, load_weights

Pair = Tuple[str, str]


def load_pairs(path: str) -> List[Pair]:
    ext = os.path.splitext(path)[1].lower()
    pairs: List[Pair] = []

    def from_obj(o: dict) -> Optional[Pair]:
        for qk, ak in (("question", "answer"), ("query", "positive"), ("q", "a")):
            if qk in o and ak in o:
                q, a = str(o[qk]).strip(), str(o[ak]).strip()
                if q and a:
                    return q, a
        return None

    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p = from_obj(json.loads(line))
                    if p:
                        pairs.append(p)
    elif ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for o in data:
            p = from_obj(o)
            if p:
                pairs.append(p)
    elif ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    pairs.append((row[0].strip(), row[1].strip()))
    else:
        raise ValueError(f"Unsupported data format: {ext}")
    if not pairs:
        raise ValueError(f"No usable (query, positive) pairs found in {path}")
    return pairs


def _encode(model: CustomEncoder, texts: List[str]) -> torch.Tensor:
    emb, mask, _ = build_token_inputs(texts)
    return model(torch.from_numpy(emb), torch.from_numpy(mask))


def finetune(
    base_path: str,
    pairs: List[Pair],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 5e-5,
    temp: float = 0.05,
    w_corr: float = 0.1,
    log: Callable[[str], None] = print,
) -> CustomEncoder:
    model = CustomEncoder()
    load_weights(model, base_path)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    n = len(pairs)
    bs = min(batch_size, n)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    steps_per_epoch = max(1, n // bs)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * bs:(s + 1) * bs]
            if len(idx) < 2:
                continue
            q = [pairs[i][0] for i in idx]
            p = [pairs[i][1] for i in idx]
            zq = _encode(model, q)
            zp = _encode(model, p)
            scores = (zq @ zp.T) / temp
            labels = torch.arange(len(idx))
            loss_nce = 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))
            cos = zq @ zp.T
            target = torch.eye(len(idx))
            loss_corr = F.mse_loss(cos, target)
            loss = loss_nce + w_corr * loss_corr
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            log(f"epoch {epoch+1}/{epochs} loss={epoch_loss/steps_per_epoch:.4f} "
                f"({time.perf_counter()-t0:.0f}s)")
    model.eval()
    return model
