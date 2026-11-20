"""Pick and freeze the untouched correctness holdout (D-018).

Unlike the diverse sample (D-012), which favoured rare features, the holdout is a plain
uniform random sample, so it stands for FineVideo as it is:
  - files: N_FILES drawn uniformly (seed SEED) from train-00050 .. train-01356, leaving out
    every file already used in the 101-video subset (and the 50 graded files);
  - videos: ROWS_PER_FILE rows drawn uniformly from each file (same seed).
The holdout is used only to confirm a chosen configuration. Nothing is tuned on it.

Usage (on the server):
  python validation/holdout/select_holdout.py files          # print the files to download
  python validation/holdout/select_holdout.py freeze --dir data/finevideo_holdout/data \
      --out validation/holdout/holdout_subset.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
from freeze_subset import sha256  # noqa: E402

SEED = 20261121
N_FILES = 8
ROWS_PER_FILE = 6
TOTAL_FILES = 1357


def chosen_files() -> list[str]:
    used = set(json.loads((ROOT / "validation/subset/frozen_subset.json").read_text())["files"])
    pool = [f"train-{i:05d}-of-01357.parquet" for i in range(50, TOTAL_FILES)]
    pool = [f for f in pool if f not in used]
    return sorted(random.Random(SEED).sample(pool, N_FILES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["files", "freeze"])
    ap.add_argument("--dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    files = chosen_files()
    if a.cmd == "files":
        print(" ".join(f"data/{f}" for f in files))
        return
    rng = random.Random(SEED + 1)
    meta, videos = {}, []
    for f in files:
        p = Path(a.dir) / f
        n = pq.ParquetFile(p).metadata.num_rows
        rows = sorted(rng.sample(range(n), min(ROWS_PER_FILE, n)))
        meta[f] = {"part": "holdout", "dir": a.dir, "rows": n, "size_bytes": p.stat().st_size,
                   "sha256": sha256(p)}
        videos += [{"video": f"{f}:{r}", "file": f, "row": r, "part": "holdout"} for r in rows]
    out = {"version": 1, "seed": SEED, "n_videos": len(videos), "files": meta, "videos": videos}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"froze {len(videos)} holdout videos from {len(files)} files -> {a.out}")


if __name__ == "__main__":
    main()
