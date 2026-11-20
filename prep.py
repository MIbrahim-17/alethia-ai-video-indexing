"""Untimed prep step: get the weights and build the TensorRT FP16 engine for this GPU (D-025).

    python prep.py

1. Download the official yolo11l.pt into weights/ if it isn't there, and check its sha256.
2. Build weights/yolo11l.engine with Ultralytics' TensorRT export: FP16, dynamic shapes,
   batch 1..32, imgsz 640. A TensorRT engine only runs on the GPU model and TensorRT version
   it was built with, so this has to run on the machine that does the timed run.
3. Write weights/yolo11l.engine.json (TensorRT version, GPU, build settings) and check that
   the engine loads and runs a batch.
run.py then uses the engine; if it's missing or doesn't load, run.py falls back to PyTorch
FP16 and says so on stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

# Never let Ultralytics install packages on its own: a missing package must fail loudly (D-025).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights" / "yolo11l.pt"
ENGINE = WEIGHTS.with_suffix(".engine")
URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt"  # D-011
SHA256 = "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"  # the weights used for every reference
# simplify=False: the optional onnxslim pass needs onnxruntime-gpu, an extra dependency, and TensorRT
# optimises the graph itself (D-025).
EXPORT_KW = dict(format="engine", quantize=16, dynamic=True, batch=32, imgsz=640, simplify=False, device=0)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def get_weights():
    WEIGHTS.parent.mkdir(exist_ok=True)
    if not WEIGHTS.exists():
        print(f"downloading {URL}")
        tmp = WEIGHTS.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(WEIGHTS)
    got = sha256(WEIGHTS)
    if got != SHA256:
        raise SystemExit(f"{WEIGHTS} sha256 {got} is not the official yolo11l.pt ({SHA256})")
    print(f"weights ok: {WEIGHTS} ({got[:12]}...)")


def build_engine(force: bool):
    import tensorrt as trt
    import torch
    from ultralytics import YOLO

    meta_path = ENGINE.with_suffix(".engine.json")
    want = {"tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(0),
            "export": {k: v for k, v in EXPORT_KW.items() if k != "device"}, "weights_sha256": SHA256}
    if ENGINE.exists() and meta_path.exists() and not force:
        have = json.loads(meta_path.read_text())
        if {k: have.get(k) for k in want} == want:
            print(f"engine up to date: {ENGINE}")
            return
    t0 = time.time()
    out = YOLO(str(WEIGHTS)).export(**EXPORT_KW)
    if Path(out).resolve() != ENGINE.resolve():
        shutil.move(out, ENGINE)
    meta_path.write_text(json.dumps({**want, "built_seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"built {ENGINE} in {time.time() - t0:.0f}s with TensorRT {trt.__version__} on {want['gpu']}")


def check_engine():
    import numpy as np
    from ultralytics import YOLO

    m = YOLO(str(ENGINE), task="detect")
    imgs = [np.full((360, 640, 3), 114, np.uint8)] * 32
    r = m.predict(imgs, imgsz=640, conf=0.4, iou=0.7, verbose=False, device=0)
    assert len(r) == 32
    print(f"engine loads and runs a batch of 32: ok ({len(m.names)} classes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="rebuild the engine even if it is up to date")
    ap.add_argument("--no-engine", action="store_true", help="only fetch the weights")
    a = ap.parse_args()
    get_weights()
    if a.no_engine:
        return
    build_engine(a.force)
    check_engine()


if __name__ == "__main__":
    sys.exit(main())
