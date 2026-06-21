#!/usr/bin/env python
"""Management CLI for the semantic FAQ backend."""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODELS_DIR = os.path.join(HERE, "app", "models")
DEFAULT_BASE = os.path.join(MODELS_DIR, "custom_hybrid.pt")
DEFAULT_NPZ = os.path.join(MODELS_DIR, "custom_hybrid_encoder.npz")
DEFAULT_ONNX = os.path.join(MODELS_DIR, "custom_hybrid_encoder.onnx")


def cmd_finetune(args: argparse.Namespace) -> None:
    from app.training.encoder_torch import export_npz, export_onnx
    from app.training.finetune import finetune, load_pairs

    base = args.base if os.path.exists(args.base) else DEFAULT_NPZ
    print(f"[finetune] base weights : {base}")
    print(f"[finetune] data         : {args.data}")
    pairs = load_pairs(args.data)
    print(f"[finetune] loaded {len(pairs)} QA pairs")

    model = finetune(
        base_path=base, pairs=pairs,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    export_npz(model, args.out_npz)
    print(f"[finetune] exported numpy weights -> {args.out_npz}")
    export_onnx(model, args.out_onnx)
    print(f"[finetune] exported ONNX          -> {args.out_onnx}")

    if args.reload:
        try:
            import httpx

            r = httpx.post(args.reload_url, timeout=30.0)
            print(f"[finetune] hot-reload {args.reload_url} -> {r.status_code} {r.text}")
        except Exception as exc:  # noqa: BLE001
            print(f"[finetune] hot-reload failed ({exc}). "
                  f"The new weights are on disk; POST {args.reload_url} or restart to apply.")
    else:
        print(f"[finetune] done. Apply with: POST {args.reload_url} (or restart the server).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic FAQ backend management CLI")
    sub = p.add_subparsers(dest="command", required=True)

    ft = sub.add_parser("finetune", help="Fine-tune the encoder on new QA pairs")
    ft.add_argument("--data", required=True, help="Path to QA pairs (.jsonl/.json/.csv/.tsv)")
    ft.add_argument("--base", default=DEFAULT_BASE, help="Existing weights (.pt or .npz)")
    ft.add_argument("--epochs", type=int, default=30)
    ft.add_argument("--batch-size", type=int, default=32)
    ft.add_argument("--lr", type=float, default=5e-5)
    ft.add_argument("--out-npz", default=DEFAULT_NPZ)
    ft.add_argument("--out-onnx", default=DEFAULT_ONNX)
    ft.add_argument("--reload", action="store_true", help="Hot-reload a running server after export")
    ft.add_argument("--reload-url", default="http://localhost:8000/admin/reload-model")
    ft.set_defaults(func=cmd_finetune)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
