"""Step 0 of D-020: profile the current pipeline (VPN, FP32 PyTorch) stage by stage.

Runs the real run.py code path (process_video_staged + Detector) on a fixed profiling set
and attributes wall time to stages. Each stage is timed where it happens, with CUDA
synchronised around predict, so the numbers add up to the wall time:
  read      Parquet row -> mp4 bytes
  decode    PyAV demux + decode (each next() on the frame generator)
  convert   frame_to_bgr (YUV -> BGR + rotation) for frames sent to the model
  predict   Detector._predict, split with Ultralytics's own per-image timers into
            preprocess (letterbox, HWC->CHW, to GPU, /255), inference (model forward),
            postprocess (NMS, scaling, Results), and the rest of predict (call overhead)
  other     everything else: sampling logic, aggregation, writing
GPU utilisation is sampled with nvidia-smi every 200 ms.

Isolated measurements:
  - decode only, every frame, with PyAV's default threading (SLICE) and with FRAME/AUTO;
  - model forward only at 640x384 FP32, batch sizes 1..64 (images per second).

Usage (GPU 0 only):
  CUDA_VISIBLE_DEVICES=0 python validation/profile_run.py --out work/profile/p0.json
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run  # noqa: E402

PROFILE_SET = ROOT / "validation" / "subset" / "profile_set.json"


def make_profile_set():
    """Every 7th sequential video of the frozen subset: 10 videos, fixed for all of D-020."""
    sub = json.loads((ROOT / "validation/subset/frozen_subset.json").read_text())
    seq = [v for v in sub["videos"] if v["part"] == "sequential"][::7]
    PROFILE_SET.write_text(json.dumps({"rule": "frozen_subset sequential videos [::7]",
                                       "videos": seq}, indent=1))
    return seq


def load_mp4s(videos):
    import pyarrow.parquet as pq
    out, t = [], 0.0
    by_file = defaultdict(set)
    for v in videos:
        by_file[v["file"]].add(v["row"])
    for f, rows in sorted(by_file.items()):
        pf = pq.ParquetFile(ROOT / "data/finevideo/data" / f)
        t0 = time.perf_counter()
        for r, b in enumerate(pf.iter_batches(batch_size=1, columns=["mp4"])):
            if r in rows:
                out.append((f"{f}:{r}", b.column("mp4")[0].as_py()))
        t += time.perf_counter() - t0
    return out, t


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.vals, self.stop = [], False
        self.p = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu",
                                   "--format=csv,noheader,nounits", "-lms", "200"],
                                  stdout=subprocess.PIPE, text=True)

    def run(self):
        for line in self.p.stdout:
            if self.stop:
                break
            try:
                self.vals.append(int(line.strip()))
            except ValueError:
                pass

    def finish(self):
        self.stop = True
        self.p.terminate()
        return sum(self.vals) / max(len(self.vals), 1)


def profile_pipeline(mp4s, det):
    import torch
    T = defaultdict(float)
    C = defaultdict(int)
    orig_iter, orig_conv, orig_pred = run.iter_raw_frames, run.frame_to_bgr, det._predict

    def timed_iter(mp4):
        g = orig_iter(mp4)
        while True:
            t0 = time.perf_counter()
            try:
                item = next(g)
            except StopIteration:
                T["decode"] += time.perf_counter() - t0
                return
            T["decode"] += time.perf_counter() - t0
            C["decoded"] += 1
            yield item

    def timed_conv(vf, k):
        t0 = time.perf_counter()
        img = orig_conv(vf, k)
        T["convert"] += time.perf_counter() - t0
        return img

    def timed_pred(imgs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = orig_pred(imgs)
        torch.cuda.synchronize()
        T["predict"] += time.perf_counter() - t0
        for r in res:  # Ultralytics's per-image timers, in ms (CUDA-synchronised)
            T["pre"] += r.speed["preprocess"] / 1e3
            T["inf"] += r.speed["inference"] / 1e3
            T["post"] += r.speed["postprocess"] / 1e3
        C["calls"] += 1
        C["inferred"] += len(imgs)
        return res

    run.iter_raw_frames, run.frame_to_bgr, det._predict = timed_iter, timed_conv, timed_pred
    w = run.Writer("/dev/null")
    gpu = GpuSampler()
    gpu.start()
    t0 = time.perf_counter()
    try:
        for vid, mp4 in mp4s:
            run.process_video_staged(vid, mp4, det, w, True, (1, 3, 0), "VPN")
    finally:
        run.iter_raw_frames, run.frame_to_bgr, det._predict = orig_iter, orig_conv, orig_pred
    wall = time.perf_counter() - t0
    util = gpu.finish()
    T["predict_overhead"] = T["predict"] - T["pre"] - T["inf"] - T["post"]
    T["other"] = wall - T["decode"] - T["convert"] - T["predict"]
    return wall, dict(T), dict(C), util


def decode_only(mp4s, thread_type):
    import av
    n, t0 = 0, time.perf_counter()
    for _, mp4 in mp4s:
        with av.open(io.BytesIO(mp4)) as c:
            s = c.streams.video[0]
            if thread_type:
                s.thread_type = thread_type
            for _ in c.decode(s):
                n += 1
    return n, time.perf_counter() - t0


def model_only(det, batches=(1, 4, 8, 16, 32, 64), iters=20):
    import torch
    from ultralytics.utils.torch_utils import select_device  # noqa: F401
    det.model.predict(__import__("numpy").zeros((360, 640, 3), "uint8"), device=0, verbose=False)
    m = det.model.predictor.model  # AutoBackend as used by predict
    out = {}
    for b in batches:
        x = torch.rand(b, 3, 384, 640, device="cuda")
        with torch.inference_mode():
            for _ in range(3):
                m(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                m(x)
            torch.cuda.synchronize()
        out[b] = round(b * iters / (time.perf_counter() - t0), 1)
    det._release_predictor_state()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/profile/p0.json")
    a = ap.parse_args()
    videos = make_profile_set()
    mp4s, t_read = load_mp4s(videos)
    det = run.Detector(ROOT / "weights" / "yolo11l.pt")
    det.detect([__import__("numpy").zeros((360, 640, 3), "uint8")])  # warm-up, not timed

    wall, T, C, util = profile_pipeline(mp4s, det)
    print(f"profiling set: {len(mp4s)} videos, {C['decoded']} frames decoded, "
          f"{C['inferred']} run ({C['inferred'] / C['decoded']:.1%}), {C['calls']} predict calls "
          f"({C['inferred'] / C['calls']:.1f} frames per call)")
    print(f"wall {wall:.1f} s (+ Parquet read {t_read:.1f} s, outside the loop); GPU util {util:.0f}%")
    rows = [("decode (PyAV)", T["decode"]), ("convert YUV->BGR", T["convert"]),
            ("predict: preprocess", T["pre"]), ("predict: inference", T["inf"]),
            ("predict: postprocess", T["post"]), ("predict: other overhead", T["predict_overhead"]),
            ("other (sampling, aggregation, writing)", T["other"])]
    for name, t in rows:
        print(f"  {name:40s} {t:8.1f} s  {t / wall:6.1%}")

    dec = {}
    for tt in (None, "FRAME", "AUTO"):
        n, t = decode_only(mp4s, tt)
        dec[tt or "default(SLICE)"] = round(n / t, 1)
    print(f"decode only, every frame, frames/s: {dec}")
    mo = model_only(det)
    print(f"model forward only, FP32 640x384, images/s by batch size: {mo}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"videos": [v for v, _ in mp4s], "wall": wall, "read": t_read,
                                       "stages": T, "counts": C, "gpu_util": util,
                                       "decode_only_fps": dec, "model_only_ips": mo}, indent=1))


if __name__ == "__main__":
    main()
