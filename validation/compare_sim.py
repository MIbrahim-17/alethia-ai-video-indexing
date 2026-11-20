"""Compare a real run.py output with the simulator's prediction for the same strategy (D-017).

Checks, line by line, that every second matches the simulator's predicted line, and,
from run.py's per-video log lines (`<video> frames=N inferred=M ...`), that the number
of frames sent to the model matches the simulator's count for each video.

Usage:
  python validation/compare_sim.py --pred work/s3/all.jsonl --sim work/sampling/sim_vpn.jsonl \
      --logs work/s3/seq.log work/s3/diverse.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(path):
    out, errs = {}, []
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            errs.append(r)
        else:
            out[(r["video"], r["second"])] = (r["objects"], r["people"])
    return out, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--logs", nargs="+", required=True)
    a = ap.parse_args()

    pred, perr = load(a.pred)
    sim, _ = load(a.sim)
    diff = [k for k in sim if pred.get(k) != sim[k]]
    extra = [k for k in pred if k not in sim]
    print(f"seconds: simulator {len(sim)}, run.py {len(pred)}, differing {len(diff)}, "
          f"extra {len(extra)}, error lines {len(perr)}")
    for k in diff[:20]:
        print(f"  DIFF {k}: sim {sim[k]} run {pred.get(k)}")

    counts = json.loads(Path(a.sim).with_suffix(".counts.json").read_text())
    got = {}
    for log in a.logs:
        for line in open(log):
            m = re.match(r"(\S+) frames=(\d+) inferred=(\d+) ", line)
            if m:
                got[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    bad = [(v, counts[v], got.get(v)) for v in counts
           if got.get(v) != (counts[v]["frames"], counts[v]["inferred"])]
    print(f"videos: simulator {len(counts)}, logged {len(got)}, frame-count mismatches {len(bad)}; "
          f"inferred total run.py {sum(x[1] for x in got.values())}, "
          f"simulator {sum(c['inferred'] for c in counts.values())}")
    for b in bad[:20]:
        print(f"  COUNT {b}")
    raise SystemExit(1 if diff or extra or bad or perr else 0)


if __name__ == "__main__":
    main()
