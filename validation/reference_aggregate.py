"""Turn the per-frame reference detections into the per-second reference JSONL (brief §5).

The rules come straight from brief §5:
  - An object is present in a second if it is detected in at least half of that
    second's frames (2 * frames_with_object >= n_frames).
  - people = the median of the per-frame person counts, rounded down when it falls
    halfway between two numbers (floor of the median).
  - There is a line for every second from 0 to the second holding the last frame. A
    second with no frames (a timestamp gap) gets objects [] and people 0 (D-014).
  - A video with an error gets its completed second lines plus one error line.

Usage:
  python validation/reference_aggregate.py --ref work/reference --out work/reference/reference.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]
PERSON = 0  # index of "person" in TARGETS


def aggregate_video(seconds: list[int], det_cls: list[list[int]]) -> list[dict]:
    per_sec: dict[int, list[list[int]]] = defaultdict(list)
    for s, cls in zip(seconds, det_cls):
        per_sec[s].append(cls)
    if not per_sec:
        return []
    out = []
    for s in range(0, max(per_sec) + 1):
        frames = per_sec.get(s, [])
        n = len(frames)
        if n == 0:
            out.append({"second": s, "objects": [], "people": 0})
            continue
        objects = [name for k, name in enumerate(TARGETS)
                   if 2 * sum(1 for f in frames if k in f) >= n]
        people = math.floor(statistics.median(f.count(PERSON) for f in frames))
        out.append({"second": s, "objects": objects, "people": people})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="output folder of build_reference.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    frames_dir = Path(a.ref) / "frames"
    n_lines = n_err = 0
    with open(a.out, "w") as fo:
        for meta_path in sorted(frames_dir.glob("*.json")):
            info = json.loads(meta_path.read_text())
            t = pq.read_table(meta_path.with_suffix(".parquet"), columns=["second", "det_cls"])
            seconds = t.column("second").to_pylist()
            if any(s < 0 for s in seconds):
                info["error"] = info.get("error") or "frame before first decoded frame"
            for rec in aggregate_video(seconds, t.column("det_cls").to_pylist()):
                fo.write(json.dumps({"video": info["video"], **rec}) + "\n")
                n_lines += 1
            if info.get("error"):
                fo.write(json.dumps({"video": info["video"], "error": info["error"]}) + "\n")
                n_err += 1
    print(f"wrote {n_lines} second lines and {n_err} error lines -> {a.out}")


if __name__ == "__main__":
    main()
