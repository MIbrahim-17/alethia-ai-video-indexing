"""Check the every-frame reference before anything else relies on it (D-003, D-014).

Checks:
  1. Completeness: every video in the frozen subset has frame and meta files, and no errors.
  2. Frame count: the number of decoded frames equals the number of packets found by the
     probe, which is an independent demux pass.
  3. Timestamps: the first frame is at t=0, timestamps increase strictly, and seconds are
     never negative.
  4. JSONL: each video has exactly floor(t_last) + 1 second lines, numbered 0..N with no
     gaps or duplicates, and uses only the exact object names.
  5. Batch equivalence (optional, needs a GPU): frames re-run one at a time through stock
     `model.predict` give the same target detections as the batched reference.

Usage:
  python validation/check_reference.py --subset validation/subset/frozen_subset.json \
      --ref work/reference --probe work/probe_seq.jsonl validation/subset/probe_diverse.jsonl \
      [--recheck-frames 20 --recheck-videos 4]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]


def safe_name(v: str) -> str:
    return v.replace(":", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--probe", nargs="+", required=True)
    ap.add_argument("--recheck-frames", type=int, default=0)
    ap.add_argument("--recheck-videos", type=int, default=4)
    ap.add_argument("--weights", default="weights/yolo11l.pt")
    a = ap.parse_args()

    subset = json.loads(Path(a.subset).read_text())
    probes = {}
    for p in a.probe:
        for line in open(p):
            r = json.loads(line)
            probes[r["video"]] = r
    fdir = Path(a.ref) / "frames"
    fails: list[str] = []
    stats = Counter()

    # Read the reference JSONL once.
    lines = defaultdict(list)
    errs = {}
    for line in open(Path(a.ref) / "reference.jsonl"):
        r = json.loads(line)
        if "error" in r:
            errs[r["video"]] = r["error"]
        else:
            lines[r["video"]].append(r)

    for v in subset["videos"]:
        vid = v["video"]
        meta_p, fr_p = fdir / f"{safe_name(vid)}.json", fdir / f"{safe_name(vid)}.parquet"
        if not meta_p.exists() or not fr_p.exists():
            fails.append(f"{vid}: missing reference files")
            continue
        info = json.loads(meta_p.read_text())
        if info.get("error") or vid in errs:
            fails.append(f"{vid}: error {info.get('error') or errs[vid]}")
        t = pq.read_table(fr_p, columns=["pts", "t_rel", "second"])
        pts = t.column("pts").to_pylist()
        secs = t.column("second").to_pylist()
        n = len(pts)
        stats["frames"] += n

        pr = probes.get(vid)
        if pr is None:
            fails.append(f"{vid}: no probe record")
        elif pr.get("n_packets") != n:
            fails.append(f"{vid}: decoded {n} frames but the probe found {pr.get('n_packets')} packets")

        if n == 0:
            fails.append(f"{vid}: zero frames")
            continue
        if t.column("t_rel")[0].as_py() != 0.0:
            fails.append(f"{vid}: first frame t_rel != 0")
        if any(b <= x for x, b in zip(pts, pts[1:])):
            fails.append(f"{vid}: pts not strictly increasing")
        if min(secs) < 0:
            fails.append(f"{vid}: negative second")

        want = max(secs) + 1
        got = [r["second"] for r in lines.get(vid, [])]
        if sorted(got) != list(range(want)):
            fails.append(f"{vid}: second lines {len(got)} (expected 0..{want - 1}, no gaps or duplicates)")
        for r in lines.get(vid, []):
            if any(o not in TARGETS for o in r["objects"]) or not isinstance(r["people"], int):
                fails.append(f"{vid}: bad record {r}")
                break
            stats["seconds"] += 1
            stats["people>0"] += r["people"] > 0
            for o in r["objects"]:
                stats[f"obj:{o}"] += 1
        empty_secs = want - len(set(secs))
        stats["seconds_without_frames"] += empty_secs

    print("== reference summary")
    for k, val in sorted(stats.items()):
        print(f"  {k:24s} {val}")

    if a.recheck_frames:
        fails += recheck_batch_equivalence(subset, fdir, a)

    print(f"== {len(fails)} failures")
    for f in fails[:50]:
        print("  FAIL", f)
    sys.exit(1 if fails else 0)


def recheck_batch_equivalence(subset, fdir, a) -> list[str]:
    """Re-run random frames one at a time through stock predict and compare to the batched reference."""
    import av
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(a.weights)
    name_to_id = {v: k for k, v in model.names.items()}
    tid = {name_to_id[n]: i for i, n in enumerate(TARGETS)}
    rng = random.Random(0)
    vids = rng.sample(subset["videos"], a.recheck_videos)
    fails, n_cmp, max_conf_diff = [], 0, 0.0
    for v in vids:
        meta = subset["files"][v["file"]]
        pf = pq.ParquetFile(Path(meta["dir"]) / v["file"])
        mp4 = None
        for r, b in enumerate(pf.iter_batches(batch_size=1, columns=["mp4"])):
            if r == v["row"]:
                mp4 = b.column("mp4")[0].as_py()
                break
        ref = pq.read_table(fdir / f"{safe_name(v['video'])}.parquet").to_pylist()
        pick = set(rng.sample(range(len(ref)), min(a.recheck_frames, len(ref))))
        with av.open(io.BytesIO(mp4)) as c:
            for i, f in enumerate(c.decode(c.streams.video[0])):
                if i not in pick:
                    continue
                img = f.to_ndarray(format="bgr24")
                rot = int(f.rotation or 0)
                if rot % 360:
                    img = np.ascontiguousarray(np.rot90(img, k=(rot // 90) % 4))
                res = model.predict(img, imgsz=640, conf=0.4, iou=0.7, verbose=False)[0]
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                got = sorted((tid[c], round(float(s), 3)) for c, s in zip(cls, conf) if c in tid)
                exp = sorted(zip(ref[i]["det_cls"], [round(x, 3) for x in ref[i]["det_conf"]]))
                n_cmp += 1
                if [g[0] for g in got] != [e[0] for e in exp]:
                    fails.append(f"{v['video']} frame {i}: single {got} vs batched {exp}")
                else:
                    for (_, g), (_, e) in zip(got, exp):
                        max_conf_diff = max(max_conf_diff, abs(g - e))
    print(f"== batch-equivalence: {n_cmp} frames compared, max conf diff {max_conf_diff:.4f}, "
          f"{len(fails)} class-set mismatches")
    return fails


if __name__ == "__main__":
    main()
