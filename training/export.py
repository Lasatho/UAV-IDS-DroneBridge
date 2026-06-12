"""
Export entry point: PyTorch checkpoint -> ONNX.

    python export.py --checkpoint checkpoints/<run>/best.pt [--opset 17]

Exports the *anomaly-score path* (window in -> score out), i.e. exactly
what runs on the Jetson, not the training graph. Dynamic batch axis.
If onnxruntime is installed, output parity against PyTorch is verified.

TensorRT engine building happens ON THE ORIN (engines are
hardware-specific), from the exported ONNX:

    # FP16
    trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16
    # INT8 (needs calibration data, see export of calibration windows below)
    trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8 \
            --calib=<calibration cache>

`--export-calibration N` additionally writes N normalized test windows
as .npy for INT8 calibration on the device.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import build_datasets, get_num_features
from models import build_model
from config import set_seed

logger = logging.getLogger("export")


class ScoreWrapper(nn.Module):
    """Exposes model.anomaly_score as a plain forward pass for tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.anomaly_score(x)


def export_onnx(model, window_length: int, num_features: int,
                out_path: Path, opset: int) -> None:
    wrapper = ScoreWrapper(model).eval()
    dummy = torch.randn(2, window_length, num_features)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["window"],
        output_names=["anomaly_score"],
        dynamic_axes={"window": {0: "batch"}, "anomaly_score": {0: "batch"}},
        opset_version=opset,
    )
    logger.info("ONNX written: %s (%.2f MB)",
                out_path, out_path.stat().st_size / 1024**2)


def verify_parity(model, onnx_path: Path, window_length: int,
                  num_features: int, atol: float = 1e-4) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping parity check. "
                       "pip install onnxruntime")
        return

    x = torch.randn(8, window_length, num_features)
    with torch.no_grad():
        ref = ScoreWrapper(model).eval()(x).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"window": x.numpy()})[0]

    max_diff = float(np.abs(ref - out).max())
    if max_diff > atol:
        raise RuntimeError(
            f"ONNX/PyTorch mismatch: max diff {max_diff:.2e} > atol {atol:.0e}"
        )
    logger.info("Parity check OK (max diff %.2e)", max_diff)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None,
                        help="Output .onnx path (default: next to checkpoint)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--export-calibration", type=int, default=0,
                        metavar="N",
                        help="Additionally export N test windows for INT8 calibration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    set_seed(int(cfg["seed"]))

    name = cfg["model"]["name"]
    window_length = int(cfg["data"]["window_length"])
    num_features = get_num_features(cfg["data"].get("features"))

    model = build_model(
        name, num_features=num_features, window_length=window_length,
        model_cfg=cfg["model"][name],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info("Loaded %s (epoch %d, val_loss %.5f)",
                name, ckpt["epoch"], ckpt["val_loss"])

    out_path = Path(args.output) if args.output else ckpt_path.with_suffix(".onnx")
    export_onnx(model, window_length, num_features, out_path, args.opset)
    verify_parity(model, out_path, window_length, num_features)

    if args.export_calibration > 0:
        datasets, _ = build_datasets(cfg["data"])
        ds = datasets["test"]
        n = min(args.export_calibration, len(ds))
        calib = torch.stack([ds[i] for i in range(n)]).numpy()
        calib_path = out_path.with_name(out_path.stem + "_calibration.npy")
        np.save(calib_path, calib)
        logger.info("Calibration windows written: %s (%d windows)",
                    calib_path, n)


if __name__ == "__main__":
    main()