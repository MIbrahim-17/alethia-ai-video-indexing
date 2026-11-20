"""Layer 2 for D-022: compare FP16 per-frame detections with the FP32 reference, frame by frame.

Both folders come from build_reference.py (same decoder, same batching); only the model
precision differs. For every frame it compares the set of target objects present, the
person count, and the confidences of detections that match by class and box (IoU >= 0.5).

Usage:
  python validation/compare_precision.py --fp32 work/reference --fp16 work/fp16/dev
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--fp16", required=True)
    a = ap.parse_args()
    c = Counter()
    per_class_flip = Counter()
    conf_diffs = []
    near = Counter()
    for meta in sorted((Path(a.fp32) / "frames").glob("*.json")):
        other = Path(a.fp16) / "frames" / meta.name
        A = pq.read_table(meta.with_suffix(".parquet")).to_pylist()
        B = pq.read_table(other.with_suffix(".parquet")).to_pylist()
        c["videos"] += 1
        if len(A) != len(B) or [r["pts"] for r in A] != [r["pts"] for r in B]:
            c["frame_list_mismatch"] += 1
            continue
        for ra, rb in zip(A, B):
            for r in (ra, rb):  # boxes are stored flat: x1, y1, x2, y2 for each box in turn
                x = r["det_xyxy"]
                if x and not isinstance(x[0], (list, tuple)):
                    r["det_xyxy"] = [x[i:i + 4] for i in range(0, len(x), 4)]
            c["frames"] += 1
            pa_, pb_ = set(ra["det_cls"]), set(rb["det_cls"])
            if pa_ != pb_:
                c["presence_differs"] += 1
                for k in pa_ ^ pb_:
                    per_class_flip[TARGETS[k]] += 1
            if ra["det_cls"].count(0) != rb["det_cls"].count(0):
                c["person_count_differs"] += 1
            if sorted(ra["det_cls"]) != sorted(rb["det_cls"]):
                c["any_box_differs"] += 1
            used = set()
            for ka, sa, ba in zip(ra["det_cls"], ra["det_conf"], ra["det_xyxy"]):
                best, bj = 0.5, None
                for j, (kb, bb) in enumerate(zip(rb["det_cls"], rb["det_xyxy"])):
                    if j not in used and kb == ka:
                        o = iou(ba, bb)
                        if o >= best:
                            best, bj = o, j
                if bj is not None:
                    used.add(bj)
                    conf_diffs.append(abs(sa - rb["det_conf"][bj]))
                elif sa < 0.45:
                    near["unmatched_fp32_box_conf_0.40-0.45"] += 1
                else:
                    near["unmatched_fp32_box_conf_>=0.45"] += 1
    d = np.array(conf_diffs)
    out = {**c, "per_class_presence_flips": dict(per_class_flip), **near,
           "matched_boxes": int(len(d)), "conf_diff_mean": float(d.mean()), "conf_diff_p99": float(np.percentile(d, 99)),
           "conf_diff_max": float(d.max())}
    f = max(c["frames"], 1)
    print(f"videos {c['videos']}, frames {c['frames']}, frame-list mismatches {c['frame_list_mismatch']}")
    print(f"frames whose object set differs: {c['presence_differs']} ({c['presence_differs'] / f:.3%}); "
          f"person count differs: {c['person_count_differs']} ({c['person_count_differs'] / f:.3%}); "
          f"any box differs: {c['any_box_differs']} ({c['any_box_differs'] / f:.3%})")
    print(f"presence flips by class: {dict(per_class_flip)}")
    print(f"matched boxes {len(d)}: |conf diff| mean {d.mean():.5f}, p99 {np.percentile(d, 99):.4f}, max {d.max():.4f}; "
          f"FP32 boxes with no FP16 match: {dict(near)}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
