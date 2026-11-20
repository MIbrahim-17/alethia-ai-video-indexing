"""Memory checks for run.py on real decoding and a real GPU (D-019). Server only.

  A. High-resolution synthetic videos (real H.264, decoded by PyAV): a 4K 30 fps video,
     a 1080p video with an oversized second (1,500 frames in one second), then the 4K video
     7 more times. run.py runs in a subprocess while its RSS is sampled every 0.2 s. The
     peak RSS of the last 4 videos must level off (within 0.15 GB), and the buffer
     peak must stay within CHUNK_BYTES + SECOND_BYTES_CAP + one frame.
  B. Frame release on real decoding: the same videos in-process, with every decoded frame
     wrapped in a weak-referenceable proxy. After each process() call, no frame of a
     written second may still be alive.
  C. CUDA OOM recovery on the real model: a GPU memory limit makes a batch of 32 fail.
     Each retry must start with exactly the GPU memory allocated before the failed
     attempt (nothing of the failed batch is still referenced), the results must equal the
     unlimited run, the predictor must hold no batch afterwards, and an OOM on a single
     frame must become a VideoError with memory back to the starting level. The old
     handler (retry inside the `except` block) is measured too, for comparison.

Usage:  python validation/test_memory.py --work work/memtest
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import subprocess
import sys
import threading
import time
import weakref
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "fixtures"))
import run  # noqa: E402

PRECISION = "fp32"
FAST = {"crf": "28", "preset": "ultrafast", "g": "30", "bf": "0"}


def make_videos(work: Path):
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq
    from make_fixtures import encode

    P = np.load(ROOT / "work" / "fixtures" / "P.npy")  # a real FineVideo frame with a person
    vids = {}
    big = cv2.resize(P, (3840, 2160), interpolation=cv2.INTER_LINEAR)
    frames = (np.roll(big, 8 * (i % 240), axis=1) for i in range(1800))  # 60 s of 4K at 30 fps
    vids["4k30_60s"] = encode(frames, [Fraction(i, 30) for i in range(1800)], size=(3840, 2160),
                              x264_opts=FAST)
    hd = cv2.resize(P, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    t = [Fraction(i, 30) for i in range(30)] + [1 + Fraction(i, 1500) for i in range(1500)] \
        + [2 + Fraction(i, 30) for i in range(90)]
    vids["1080p_dense_second"] = encode((np.roll(hd, 4 * i, axis=1) for i in range(len(t))), t,
                                        size=(1920, 1080), x264_opts=FAST)
    order = ["4k30_60s", "1080p_dense_second"] + ["4k30_60s"] * 7
    (work / "input").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"mp4": pa.array([vids[k] for k in order], pa.binary()),
                             "json": pa.array(["{}"] * len(order), pa.string())}),
                   work / "input" / "memtest.parquet")
    return order


def rss_kb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0


def part_a(work: Path, order):
    p = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--input", str(work / "input"),
                          "--output", str(work / "out.jsonl"), "--precision", PRECISION], stderr=subprocess.PIPE, text=True,
                         env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"})
    samples, stop = [], False

    def sampler():
        while not stop:
            samples.append((time.time(), rss_kb(p.pid)))
            time.sleep(0.2)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    marks, done = [time.time()], ""
    for line in p.stderr:
        if line.startswith("memtest.parquet:"):
            marks.append(time.time())
            print("   ", line.strip())
        if line.startswith("DONE"):
            done = line.strip()
    p.wait()
    stop = True
    th.join()
    per_video = []
    for k in range(len(order)):
        seg = [r for ts, r in samples if marks[k] <= ts <= marks[k + 1]]
        per_video.append(max(seg) / 2**20 if seg else 0.0)
    print("    " + done)
    stats = dict(kv.split("=") for kv in done.split()[1:] if "=" in kv)
    bound_mb = (run.CHUNK_BYTES + run.SECOND_BYTES_CAP + 3840 * 2160 * 3 // 2) / 2**20
    four_k = [per_video[i] for i, k in enumerate(order) if k.startswith("4k")]
    ok = (p.returncode == 0 and int(stats["errors"]) == 0
          and float(stats["peak_buffered_mb"]) <= bound_mb
          and int(stats["overflow_seconds"]) == 1
          and max(four_k[-4:]) - min(four_k[-4:]) <= 0.15)  # levels off: no growth per video
    print(f"  A: peak RSS per video (GB) {[round(x, 2) for x in per_video]}; "
          f"buffer peak {stats['peak_buffered_mb']} MB (bound {bound_mb:.0f}); "
          f"overflow seconds {stats['overflow_seconds']} -> {'PASS' if ok else 'FAIL'}")
    return ok


class Proxy:
    """A weak-referenceable stand-in for a PyAV VideoFrame."""
    __slots__ = ("f", "__weakref__")

    def __init__(self, f):
        self.f = f

    @property
    def width(self):
        return self.f.width

    @property
    def height(self):
        return self.f.height

    def to_ndarray(self, **kw):
        return self.f.to_ndarray(**kw)


def part_b(work: Path, det):
    import pyarrow.parquet as pq

    mp4s = pq.read_table(work / "input" / "memtest.parquet").column("mp4").to_pylist()[:2]
    orig_iter, orig_proc = run.iter_raw_frames, run.StagedVideo.process
    ok = True
    for name, mp4 in zip(["4k30_60s", "1080p_dense_second"], mp4s):
        refs, violations = {}, []

        def wrapped(m):
            for t, f, k in orig_iter(m):
                px = Proxy(f)
                refs.setdefault(int(t // 1), []).append(weakref.ref(px))
                yield t, px, k

        def tracked(self, final):
            orig_proc(self, final)
            alive_from = min([x.s for x in self.pending] + ([self.cur.s] if self.cur else []) + [10**9])
            violations.extend(s for s, rs in refs.items()
                              if s < alive_from and any(r() is not None for r in rs))

        run.iter_raw_frames, run.StagedVideo.process = wrapped, tracked
        w = run.Writer(str(work / f"b_{name}.jsonl"))
        try:
            run.process_video_staged(name, mp4, det, w, True, (1, 3, 0), "VPN")
        finally:
            run.iter_raw_frames, run.StagedVideo.process = orig_iter, orig_proc
            w.close()
        gc.collect()
        left = sum(r() is not None for rs in refs.values() for r in rs)
        n = sum(len(rs) for rs in refs.values())
        good = not violations and left == 0
        ok &= good
        print(f"  B: {name}: {n} frames decoded, written seconds with live frames: "
              f"{sorted(set(violations))[:5]}, frames alive after the video: {left} -> "
              f"{'PASS' if good else 'FAIL'}")
    return ok


def part_c(work: Path, det):
    import pyarrow.parquet as pq
    import torch

    mp4 = pq.read_table(work / "input" / "memtest.parquet").column("mp4").to_pylist()[0]
    imgs = [img for _, (_, img) in zip(range(32), run.iter_frames(mp4))]  # 32 real 4K frames
    det.batch_cap.clear()
    want = det.detect(imgs)                       # no limit
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    det.detect(imgs[:4])
    need4 = torch.cuda.max_memory_allocated() - base
    total = torch.cuda.get_device_properties(0).total_memory
    ok = True

    def limited(extra, fn):
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(min(1.0, (base + extra) / total), 0)
        try:
            return fn()
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)

    # New handler.
    entries = []
    orig = det._predict

    def spy(batch):
        entries.append((len(batch), torch.cuda.memory_allocated()))
        return orig(batch)

    det._predict, det.batch_cap = spy, {}
    oom0 = det.n_oom
    got = limited(int(need4 * 1.3), lambda: det.detect(imgs))
    det._predict = orig
    after = torch.cuda.memory_allocated()
    p = det.model.predictor
    held = (p.batch, p.results, p.dataset)
    good = (got == want and det.n_oom > oom0 and all(m == base for _, m in entries)
            and after == base and held == (None, None, None))
    ok &= good
    print(f"  C: new handler: attempts (batch size, MB allocated at start) "
          f"{[(n, round(m / 2**20)) for n, m in entries]}; base {base / 2**20:.0f} MB; "
          f"after {after / 2**20:.0f} MB; OOMs {det.n_oom - oom0}; results equal: {got == want}; "
          f"predictor holds a batch: {held != (None, None, None)} -> {'PASS' if good else 'FAIL'}")

    # Old handler (D-015 C5 as first written): retry inside the except block.
    old_entries = []

    def old_detect(batch):
        old_entries.append((len(batch), torch.cuda.memory_allocated()))
        try:
            return det.model.predict(batch, device=0, **run.PREDICT_KW,
                                     **({"quantize": 16} if det.half else {}))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise run.VideoError("old handler: OOM on a single frame")
            mid = len(batch) // 2
            return old_detect(batch[:mid]) + old_detect(batch[mid:])

    try:
        limited(int(need4 * 1.3), lambda: old_detect(imgs))
        old_outcome = "recovered"
    except run.VideoError as e:
        old_outcome = f"FAILED ({e})"
    det._release_predictor_state()
    gc.collect()
    print(f"     old handler, for comparison: attempts (batch size, MB at start) "
          f"{[(n, round(m / 2**20)) for n, m in old_entries[:6]]} ... "
          f"max extra at a retry start {max(m for _, m in old_entries) / 2**20 - base / 2**20:.0f} MB; "
          f"{len(old_entries)} attempts; outcome: {old_outcome}")

    # Single-frame OOM: a VideoError, memory back to base, and the detector still works afterwards.
    det.batch_cap = {}
    err = None
    try:
        limited(int(need4 * 0.05), lambda: det.detect(imgs[:1]))
    except run.VideoError as e:
        err = str(e)
    det.batch_cap = {}
    again = det.detect(imgs[:8])
    after1 = torch.cuda.memory_allocated()
    good = err is not None and after1 == base and again == want[:8]
    ok &= good
    print(f"  C: single-frame OOM -> {err!r}; memory after {after1 / 2**20:.0f} MB (base "
          f"{base / 2**20:.0f}); detector works afterwards: {again == want[:8]} -> "
          f"{'PASS' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/memtest")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    a = ap.parse_args()
    global PRECISION
    PRECISION = a.precision
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    t = time.time()
    order = make_videos(work)
    print(f"made {len(order)} synthetic videos in {time.time() - t:.0f}s")
    ok_a = part_a(work, order)
    det = run.Detector(ROOT / "weights" / "yolo11l.pt", half=a.precision == "fp16")
    ok_b = part_b(work, det)
    ok_c = part_c(work, det)
    print(f"== A {'PASS' if ok_a else 'FAIL'}, B {'PASS' if ok_b else 'FAIL'}, C {'PASS' if ok_c else 'FAIL'}")
    sys.exit(0 if ok_a and ok_b and ok_c else 1)


if __name__ == "__main__":
    main()
