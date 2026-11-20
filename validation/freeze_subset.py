"""Freeze the 101-video validation subset (D-006, D-012) into one manifest.

The manifest records every video ID plus the size and sha256 of each source Parquet
file. Later runs can then check that they used exactly the same data.

Usage:
  python validation/freeze_subset.py --seq-dir data/finevideo/data \
      --diverse-dir data/finevideo_diverse/data \
      --diverse-manifest validation/subset/diverse_manifest.json \
      --out validation/subset/frozen_subset.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--diverse-dir", required=True)
    ap.add_argument("--diverse-manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files, videos = {}, []
    # Sequential part: every row of every file in seq-dir.
    for p in sorted(Path(a.seq_dir).glob("*.parquet")):
        n = pq.ParquetFile(p).metadata.num_rows
        files[p.name] = {"part": "sequential", "dir": a.seq_dir, "rows": n}
        videos += [{"video": f"{p.name}:{r}", "file": p.name, "row": r, "part": "sequential"}
                   for r in range(n)]
    # Diverse part: the rows chosen in diverse_manifest.json.
    dm = json.loads(Path(a.diverse_manifest).read_text())
    for v in sorted(dm["videos"], key=lambda v: (v["file"], v["row"])):
        files.setdefault(v["file"], {"part": "diverse", "dir": a.diverse_dir,
                                     "rows": pq.ParquetFile(Path(a.diverse_dir) / v["file"]).metadata.num_rows})
        videos.append({"video": v["video"], "file": v["file"], "row": v["row"], "part": "diverse"})

    for name, meta in files.items():
        p = Path(meta["dir"]) / name
        meta["size_bytes"] = p.stat().st_size
        meta["sha256"] = sha256(p)

    out = {"version": 1, "n_videos": len(videos), "files": files, "videos": videos}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"froze {len(videos)} videos from {len(files)} files -> {a.out}")


if __name__ == "__main__":
    main()
