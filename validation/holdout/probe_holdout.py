"""Probe the holdout videos (D-018) with the same probe_video used for the subset (D-012).

Usage: python validation/holdout/probe_holdout.py --subset validation/holdout/holdout_subset.json \
           --out work/probe_holdout.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
from select_diverse_sample import probe_video  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sub = json.loads(Path(a.subset).read_text())
with open(a.out, "w") as fo:
    for fname, meta in sub["files"].items():
        rows = {v["row"] for v in sub["videos"] if v["file"] == fname}
        for r, b in enumerate(pq.ParquetFile(Path(meta["dir"]) / fname).iter_batches(batch_size=1, columns=["mp4"])):
            if r in rows:
                fo.write(json.dumps({"video": f"{fname}:{r}", **probe_video(b.column("mp4")[0].as_py())}) + "\n")
print("probed", sum(1 for _ in open(a.out)))
