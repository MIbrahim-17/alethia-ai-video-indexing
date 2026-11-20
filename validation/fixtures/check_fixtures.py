"""Run the D-013 Critical fixtures through run.py and compare against independently derived expectations.

How the expectation for an "oracle" fixture is built, with no code shared with run.py:
  - decoding: cv2.VideoCapture, an FFmpeg build separate from PyAV, with OpenCV's own
    automatic rotation;
  - timestamps: the times the fixture was built with, measured from the first frame;
  - detection: stock single-frame `model.predict` (conf 0.4, iou 0.7, imgsz 640);
  - aggregation: validation/reference_aggregate.aggregate_video.
It also checks that each fixture is valid: the container's packet timestamps must equal
the intended ones. For rotation it also compares pixels: run.py's decoded frames
against cv2's frames.

Usage (on the server):
  python validation/fixtures/check_fixtures.py --fixtures work/fixtures [--skip-run]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT))
from reference_aggregate import TARGETS, aggregate_video  # noqa: E402
from simulate_sampling import select_staged  # noqa: E402


def cv2_frames(mp4: bytes):
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        tf.write(mp4)
        tf.flush()
        cap = cv2.VideoCapture(tf.name, cv2.CAP_FFMPEG)
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
    return frames


def container_times(mp4: bytes):
    import av

    with av.open(io.BytesIO(mp4)) as c:
        s = c.streams.video[0]
        tb = Fraction(s.time_base.numerator, s.time_base.denominator)
        return sorted(p.pts * tb for p in c.demux(s) if p.pts is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolo11l.pt"))
    ap.add_argument("--sampling", default="all", help="run.py --sampling; the oracle applies the "
                    "simulator's selection rule (validation/simulate_sampling.py) for staged strategies")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="run.py --precision; the oracle's single-frame predict uses the same precision")
    a = ap.parse_args()
    q = {"quantize": 16} if a.precision == "fp16" else {}
    stage_rule = None
    if a.sampling != "all":
        m = re.fullmatch(r"staged-([0-9a-z>]+)-([VPCN]+)", a.sampling)
        stage_rule = (tuple(0 if x == "all" else int(x) for x in m.group(1).split(">")), m.group(2))

    fdir = Path(a.fixtures)
    manifest = json.loads((fdir / "manifest.json").read_text())
    tag = a.sampling.replace(">", "_") + ("" if a.precision == "fp32" else "_fp16")
    out_path = fdir / f"run_output_{tag}.jsonl"
    if not a.skip_run:
        r = subprocess.run([sys.executable, str(ROOT / "run.py"), "--input", str(fdir / "input"),
                            "--output", str(out_path), "--sampling", a.sampling,
                            "--precision", a.precision],
                           capture_output=True, text=True)
        (fdir / f"run_stderr_{tag}.log").write_text(r.stderr)
        if r.returncode != 0:
            print(r.stderr[-3000:])
            raise SystemExit(f"run.py exited {r.returncode}")

    got_secs, got_err = defaultdict(list), defaultdict(list)
    for line in open(out_path):
        rec = json.loads(line)
        (got_err if "error" in rec else got_secs)[rec["video"]].append(rec)

    from ultralytics import YOLO

    import run as runmod

    model = YOLO(a.weights)
    tid = {k: TARGETS.index(v) for k, v in model.names.items() if v in TARGETS}
    table = pq.read_table(fdir / "input" / "fixtures-critical.parquet", columns=["mp4"])
    mp4s = table.column("mp4").to_pylist()

    results = []
    for m in manifest:
        vid, kind = m["video"], m["kind"]
        mp4 = mp4s[m["row"]]
        got = sorted(got_secs[vid], key=lambda r: r["second"])
        errs = got_err[vid]
        status, detail = "PASS", ""
        if kind == "error_only":
            if got or len(errs) != 1:
                status, detail = "FAIL", f"{len(got)} second lines, {len(errs)} error lines"
            else:
                detail = errs[0]["error"]
        elif kind == "observe":
            status = "OBSERVE"
            detail = f"{len(got)} second lines, errors={[e['error'] for e in errs]}"
        else:  # oracle, or oracle_partial (cv2 decodes only a prefix; exactly one error line is expected)
            intended = [Fraction(t) for t in m["times"]]
            ct = container_times(mp4)
            frames = cv2_frames(mp4)
            partial = kind == "oracle_partial"
            if partial:
                if not 0 < len(frames) < len(intended):
                    status, detail = "FIXTURE-INVALID", f"cv2 decoded {len(frames)} of {len(intended)}"
                intended = intended[: len(frames)]
                ct = ct[: len(frames)]
            if status != "PASS":
                pass
            elif ct != intended:
                status, detail = "FIXTURE-INVALID", f"container times {ct[:5]}... != intended"
            elif len(frames) != len(intended):
                status, detail = "FIXTURE-INVALID", f"cv2 decoded {len(frames)} of {len(intended)}"
            else:
                t0 = intended[0]
                secs = [math.floor(t - t0) for t in intended]
                dets = []
                for img in frames:
                    r = model.predict(img, imgsz=640, conf=0.4, iou=0.7, verbose=False, **q)[0]
                    dets.append([tid[int(c)] for c in r.boxes.cls.tolist() if int(c) in tid])
                if stage_rule is not None:  # keep only the frames the staged rule picks
                    keep = select_staged(secs, dets, *stage_rule)
                    secs, dets = [secs[i] for i in keep], [dets[i] for i in keep]
                exp = [{"video": vid, **e} for e in aggregate_video(secs, dets)]
                want_errs = 1 if partial else 0
                if got != exp or len(errs) != want_errs:
                    status = "FAIL"
                    detail = f"expected {exp}\n      got {got} errors={errs}"
                else:
                    detail = f"{len(exp)} seconds, people={[e['people'] for e in exp]}"
                if m["name"].startswith("rotation_"):  # pixel check of run.py's decoder vs cv2
                    ours = [img for _, img in runmod.iter_frames(mp4)]
                    if ours[0].shape != frames[0].shape:
                        status = "FAIL"
                        detail += f" | shape ours {ours[0].shape} vs cv2 {frames[0].shape}"
                    else:
                        d_ok = float(np.mean(np.abs(ours[0].astype(int) - frames[0].astype(int))))
                        alts = [float(np.mean(np.abs(np.rot90(ours[0], k).astype(int)
                                                     - frames[0].astype(int))))
                                for k in (1, 2, 3) if np.rot90(ours[0], k).shape == frames[0].shape]
                        detail += f" | pixel diff vs cv2 {d_ok:.2f} (other turns {[round(x, 1) for x in alts]})"
                        if alts and d_ok >= min(alts):
                            status = "FAIL"
        results.append((m["name"], status, detail, m.get("note", "")))

    width = max(len(n) for n, *_ in results)
    for name, status, detail, note in results:
        print(f"{status:15s} {name:{width}s}  {detail}" + (f"   [{note}]" if status == "OBSERVE" else ""))
    n_fail = sum(s in ("FAIL", "FIXTURE-INVALID") for _, s, _, _ in results)
    print(f"== {len(results)} fixtures, {n_fail} failing")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
