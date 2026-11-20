"""Explain differences between a real run.py run and the simulator (D-020 completeness rule).

Re-runs run.py's staged path in-process on the given videos and records, for every frame
it sends to the model, the detections it got (classes and confidences). These are compared
with the stored per-frame detections that the simulator used (build_reference.py output).
For each frame whose target-object set or person count differs, it prints both sets of
detections, so it's visible whether the difference is a detection near the 0.4 threshold.

Usage:
  CUDA_VISIBLE_DEVICES=0 python validation/explain_frame_diffs.py --frames work/fp16/dev \
      --precision fp16 --videos train-00000-of-01357.parquet:28 ...
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
import run  # noqa: E402
from build_reference import safe_name  # noqa: E402

DATA_DIRS = [ROOT / "data/finevideo/data", ROOT / "data/finevideo_diverse/data", ROOT / "data/finevideo_holdout/data"]


def load_mp4(video):
    f, r = video.rsplit(":", 1)
    for d in DATA_DIRS:
        if (d / f).exists():
            for i, b in enumerate(pq.ParquetFile(d / f).iter_batches(batch_size=1, columns=["mp4"])):
                if i == int(r):
                    return b.column("mp4")[0].as_py()
    raise SystemExit(f"{video} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="build_reference.py output used by the simulator")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--videos", nargs="+", required=True)
    a = ap.parse_args()
    det = run.Detector(ROOT / "weights" / "yolo11l.pt", half=a.precision == "fp16")
    seen = []  # (cls list, conf list) per image, in the order the model saw them
    orig_sum = det._summarise

    def summarise(r):
        seen.append((r.boxes.cls.int().tolist(), [round(x, 4) for x in r.boxes.conf.tolist()]))
        return orig_sum(r)

    det._summarise = summarise
    order = []  # (second, position) per image, in the same order
    orig_run = run._run_positions

    def run_positions(d, todo):
        order.extend((sec.s, p) for sec, p in todo)
        return orig_run(d, todo)

    run._run_positions = run_positions
    for video in a.videos:
        seen.clear()
        order.clear()
        w = run.Writer("/dev/null")
        run.process_video_staged(video, load_mp4(video), det, w, True, (1, 3, 0), "VPN")
        ref = pq.read_table(Path(a.frames) / "frames" / f"{safe_name(video)}.parquet").to_pylist()
        first = {}
        for i, r in enumerate(ref):
            first.setdefault(r["second"], i)
        tgt = det.target_of
        n_diff, near = 0, 0
        lines = []
        for (s, p), (cls, conf) in zip(order, seen):
            i = first[s] + p
            got = [(tgt[c], cf) for c, cf in zip(cls, conf) if c in tgt]
            g_cls = sorted(k for k, _ in got)
            want = sorted(ref[i]["det_cls"])
            if set(g_cls) != set(want) or g_cls.count(0) != want.count(0):
                n_diff += 1
                changed = set(g_cls) ^ set(want) or {0}
                cf_all = [cf for k, cf in got if k in changed] + \
                         [cf for k, cf in zip(ref[i]["det_cls"], ref[i]["det_conf"]) if k in changed]
                close = min(abs(x - 0.4) for x in cf_all) if cf_all else 1.0
                near += close < 0.02
                lines.append(f"    frame {i} (second {s}, position {p}): run.py "
                             f"{[(run.TARGETS[k], cf) for k, cf in sorted(got)]} | simulator's "
                             f"{[(run.TARGETS[k], round(c, 4)) for k, c in sorted(zip(ref[i]['det_cls'], ref[i]['det_conf']))]}"
                             f"\n      run.py, all 80 classes: {[(det.model.names[c], cf) for c, cf in zip(cls, conf)]}")
        print(f"{video}: {len(order)} frames run, {n_diff} differ from the stored detections, "
              f"{near} of them with the changed detection within 0.02 of the 0.4 threshold")
        print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
