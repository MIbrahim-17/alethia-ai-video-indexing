"""Check a full run's results.jsonl for completeness, independently of run.py (D-026).

Expected seconds come from demuxing each video's packet timestamps (no decoding): a second
is expected if at least one packet's presentation time, relative to the earliest one, falls
in it. Checks:
  - every line parses and has the required schema; no error lines;
  - every row of every input Parquet file appears as a video, and no other video does;
  - each video's lines are contiguous, with seconds strictly increasing (no duplicates);
  - each video's seconds equal the expected set from the packet timestamps;
  - (with --log) run.py's per-video decoded frame count equals the packet count.

Usage: python validation/check_full_results.py --results R.jsonl --input DIR [--log run.err] [--workers 8]
"""

import argparse
import io
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]


def expected_for_file(path):
    import av
    import pyarrow.parquet as pq

    out = {}
    pf = pq.ParquetFile(path)
    row = 0
    for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
        for mp4 in batch.column("mp4").to_pylist():
            vid = f"{Path(path).name}:{row}"
            row += 1
            try:
                with av.open(io.BytesIO(mp4), mode="r") as c:
                    st = c.streams.video[0]
                    pts = []
                    for p in c.demux(st):
                        if p.pts is None or p.size == 0:
                            continue
                        tb = p.time_base or st.time_base
                        pts.append(p.pts * Fraction(tb.numerator, tb.denominator))
                t0 = min(pts)
                out[vid] = (sorted({math.floor(t - t0) for t in pts}), len(pts), None)
            except Exception as e:  # noqa: BLE001
                out[vid] = (None, 0, f"{type(e).__name__}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--log")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    problems = []
    seen = {}  # video -> list of seconds, in file order
    order = []
    bad_schema = err_lines = 0
    with open(a.results) as f:
        for n, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad_schema += 1
                continue
            if "error" in r:
                err_lines += 1
                continue
            ok = (set(r) == {"video", "second", "objects", "people"} and isinstance(r["video"], str)
                  and isinstance(r["second"], int) and r["second"] >= 0
                  and isinstance(r["people"], int) and r["people"] >= 0
                  and isinstance(r["objects"], list)
                  and r["objects"] == [t for t in TARGETS if t in r["objects"]])
            if not ok:
                bad_schema += 1
            v = r["video"]
            if v not in seen:
                seen[v] = []
                order.append(v)
            elif order[-1] != v:
                problems.append(f"line {n}: {v} lines not contiguous")
            seen[v].append(r["second"])

    files = sorted(Path(a.input).glob("*.parquet"))
    with ProcessPoolExecutor(a.workers) as ex:
        expected = {}
        for d in ex.map(expected_for_file, [str(p) for p in files]):
            expected.update(d)

    logged = {}
    if a.log:
        for line in open(a.log):
            m = re.match(r"(\S+\.parquet:\d+) frames=(\d+) ", line)
            if m:
                logged[m.group(1)] = int(m.group(2))

    missing = [v for v in expected if v not in seen]
    extra = [v for v in seen if v not in expected]
    not_increasing = [v for v, s in seen.items() if any(b <= a_ for a_, b in zip(s, s[1:]))]
    sec_mismatch, demux_fail, frame_mismatch = [], [], []
    for v, (secs, npk, err) in expected.items():
        if err:
            demux_fail.append(f"{v}: {err}")
            continue
        if v in seen and seen[v] != secs:
            sec_mismatch.append(f"{v}: expected {len(secs)} seconds, got {len(seen[v])}; "
                                f"missing {sorted(set(secs) - set(seen[v]))[:5]} extra {sorted(set(seen[v]) - set(secs))[:5]}")
        if a.log and logged.get(v) != npk:
            frame_mismatch.append(f"{v}: packets {npk}, run.py frames {logged.get(v)}")

    total = sum(len(s) for s in seen.values())
    exp_total = sum(len(s) for s, _, e in expected.values() if not e)
    print(f"files {len(files)}  videos expected {len(expected)}  in results {len(seen)}")
    print(f"result lines {total}  expected seconds {exp_total}  error lines {err_lines}  bad schema {bad_schema}")
    print(f"missing videos {len(missing)}  extra videos {len(extra)}  non-increasing/duplicate {len(not_increasing)}")
    print(f"second-set mismatches {len(sec_mismatch)}  demux failures {len(demux_fail)}"
          + (f"  frame-count mismatches {len(frame_mismatch)} (log videos {len(logged)})" if a.log else ""))
    for lst in (problems, missing, extra, not_increasing, sec_mismatch, demux_fail, frame_mismatch):
        for x in lst[:10]:
            print("  ", x)
    ok = not (problems or missing or extra or not_increasing or sec_mismatch or demux_fail
              or frame_mismatch or err_lines or bad_schema)
    print("COMPLETE=" + str(ok))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
