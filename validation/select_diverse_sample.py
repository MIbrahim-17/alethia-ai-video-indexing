"""Probe every video in a set of FineVideo Parquet files and pick a diverse sample.

The script reads the container and stream properties from the mp4 bytes with PyAV.
That covers codec, resolution, rotation, frame rate, whether the timing is variable,
duration and the first timestamp. The `json` metadata is recorded only so it can be
compared against the probe. The script then picks videos with a greedy pass so that
every rare attribute value is covered, and fills the remaining slots with videos from
the files that are least represented so far.

Usage:
  python validation/select_diverse_sample.py --input DIR [DIR ...] --n 35 \
      --probe-out validation/subset/probe_diverse.jsonl \
      --manifest-out validation/subset/diverse_manifest.json
  # Probe only, without picking a sample:
  python validation/select_diverse_sample.py --input DIR --probe-out probe.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import pyarrow.parquet as pq


# ---------------------------------------------------------------- probing

def _rotation(stream, first_frame) -> int:
    """Return the rotation in degrees, read from the frame's display matrix or the stream tag."""
    rot = None
    if first_frame is not None:
        rot = getattr(first_frame, "rotation", None)
    if not rot:
        tag = (stream.metadata or {}).get("rotate")
        if tag is not None:
            try:
                rot = int(float(tag))
            except ValueError:
                rot = None
    return int(rot or 0) % 360


def probe_video(mp4: bytes) -> dict:
    """Read the container and stream properties without decoding every frame.

    Only the first frame is decoded. All packets are demuxed to collect timestamps.
    """
    out: dict = {"size_bytes": len(mp4)}
    try:
        with av.open(io.BytesIO(mp4), mode="r") as c:
            out["format"] = c.format.name
            vs = [s for s in c.streams if s.type == "video"]
            out["n_video_streams"] = len(vs)
            out["has_audio"] = any(s.type == "audio" for s in c.streams)
            if not vs:
                out["probe_error"] = "no video stream"
                return out
            s = vs[0]
            cc = s.codec_context
            out.update(
                codec=cc.name,
                profile=getattr(cc, "profile", None),
                pix_fmt=cc.pix_fmt,
                width=cc.width,
                height=cc.height,
                avg_fps=float(s.average_rate) if s.average_rate else None,
                base_fps=float(s.base_rate) if s.base_rate else None,
                time_base=str(s.time_base),
                stream_frames=s.frames or None,
                container_duration_s=(c.duration / 1e6) if c.duration else None,
            )

            # Collect the timestamps of every video packet, sorted into presentation order.
            pts = []
            for pkt in c.demux(s):
                if pkt.pts is not None:
                    pts.append(pkt.pts)
            pts.sort()
            out["n_packets"] = len(pts)
            if len(pts) >= 2:
                tb = float(s.time_base)
                d = [(b - a) * tb for a, b in zip(pts, pts[1:])]
                med = statistics.median(d)
                out["first_pts_s"] = pts[0] * tb
                out["span_s"] = (pts[-1] - pts[0]) * tb
                out["median_frame_dt_s"] = med
                out["max_frame_dt_s"] = max(d)
                # Variable frame rate: more than 1% of frame gaps differ from the median gap by over 10%.
                off = sum(1 for x in d if abs(x - med) > 0.1 * med)
                out["vfr_fraction"] = off / len(d)
                out["dup_pts"] = len(pts) - len(set(pts))

        # Decode the first frame in a separate pass to read its rotation and confirm decoding works.
        with av.open(io.BytesIO(mp4), mode="r") as c:
            s = c.streams.video[0]
            first = None
            try:
                for f in c.decode(s):
                    first = f
                    break
            except av.FFmpegError as e:  # decode failure is itself a useful attribute
                out["first_decode_error"] = str(e)[:200]
            out["rotation"] = _rotation(s, first)
            out["first_frame_ok"] = first is not None
    except Exception as e:  # noqa: BLE001 - any failure is recorded, not raised
        out["probe_error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def _probe_row(args) -> dict:
    path, row, mp4, meta_raw = args
    rec = {"video": f"{Path(path).name}:{row}", "file": Path(path).name, "row": row}
    try:
        meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw or "{}")
    except (TypeError, ValueError):
        meta = {}
    rec["meta_fps"] = (meta.get("content_metadata") or {}).get("fps")
    rec["meta_resolution"] = meta.get("resolution")
    rec["meta_duration_s"] = meta.get("duration_seconds")
    if mp4 is None:
        rec["probe_error"] = "null mp4"
        return rec
    rec.update(probe_video(mp4))
    return rec


def iter_rows(paths):
    for p in paths:
        pf = pq.ParquetFile(p)
        row = 0
        for batch in pf.iter_batches(batch_size=4, columns=["mp4", "json"]):
            mp4s = batch.column("mp4").to_pylist()
            metas = batch.column("json").to_pylist()
            for m, j in zip(mp4s, metas):
                yield (str(p), row, m, j)
                row += 1


# ---------------------------------------------------------------- features

def features(r: dict) -> dict:
    """Group each probe result into coarse values used for coverage."""
    if r.get("probe_error") or not r.get("first_frame_ok", False):
        return {"status": "error"}
    w, h = r["width"], r["height"]
    rot = r.get("rotation", 0)
    dw, dh = (h, w) if rot in (90, 270) else (w, h)  # size as displayed, after rotation
    short = min(dw, dh)
    fps = r.get("avg_fps") or 0
    dur = r.get("span_s") or r.get("container_duration_s") or 0
    return {
        "status": "ok",
        "codec": r["codec"],
        "pix_fmt": r["pix_fmt"],
        "res": ("<=360p" if short <= 360 else "480p" if short <= 480 else "720p"
                if short <= 720 else "1080p" if short <= 1080 else ">1080p"),
        "orientation": "portrait" if dh > dw else "square" if dh == dw else "landscape",
        "rotation": f"rot{rot}",
        "fps": ("<20" if fps < 20 else "20-26" if fps < 26.5 else "26.5-31"
                if fps < 31 else "31-50" if fps < 50 else ">=50"),
        "vfr": "vfr" if r.get("vfr_fraction", 0) > 0.01 else "cfr",
        "start": "nonzero_start" if (r.get("first_pts_s") or 0) > 1e-6 else "zero_start",
        "duration": ("<60s" if dur < 60 else "60-180s" if dur < 180 else "180-600s"
                     if dur < 600 else ">=600s"),
    }


def select(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    feats = [features(r) for r in records]
    freq = Counter((k, v) for f in feats for k, v in f.items())
    covered: set = set()
    chosen: list[int] = []
    pool = list(range(len(records)))
    rng.shuffle(pool)  # random tie-breaking, repeatable because of the seed

    # Step 1: greedy coverage. Pick the video whose not-yet-covered values are rarest.
    while len(chosen) < n:
        def gain(i):
            return sum(1.0 / freq[kv] for kv in feats[i].items() if kv not in covered)
        best = max((i for i in pool if i not in chosen), key=gain, default=None)
        if best is None or gain(best) == 0:
            break
        chosen.append(best)
        covered.update(feats[best].items())
        records[best]["selected_for"] = "coverage"

    # Step 2: fill the remaining slots, preferring files with the fewest picks so far.
    per_file = Counter(records[i]["file"] for i in chosen)
    rest = [i for i in pool if i not in chosen]
    while len(chosen) < n and rest:
        rest.sort(key=lambda i: per_file[records[i]["file"]])
        i = rest.pop(0)
        chosen.append(i)
        per_file[records[i]["file"]] += 1
        records[i]["selected_for"] = "spread"

    out = []
    for i in chosen:
        rec = dict(records[i])
        rec["features"] = feats[i]
        out.append(rec)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="folders of .parquet files")
    ap.add_argument("--probe-out", required=True)
    ap.add_argument("--manifest-out")
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    a = ap.parse_args()

    paths = sorted(p for d in a.input for p in Path(d).rglob("*.parquet"))
    print(f"probing {len(paths)} parquet files")
    records = []
    with ProcessPoolExecutor(a.workers) as ex:
        for p in paths:  # one file at a time, so at most one file's bytes are in memory
            records.extend(ex.map(_probe_row, iter_rows([p]), chunksize=1))
            print(f"  {p.name}: done", flush=True)
    records.sort(key=lambda r: (r["file"], r["row"]))

    Path(a.probe_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.probe_out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"probed {len(records)} videos -> {a.probe_out}")

    feats = [features(r) for r in records]
    for key in ["status", "codec", "pix_fmt", "res", "orientation", "rotation", "fps", "vfr",
                "start", "duration"]:
        print(f"  {key:12s} {dict(Counter(f.get(key, '-') for f in feats))}")

    if a.manifest_out:
        chosen = select(records, a.n, a.seed)
        manifest = {
            "seed": a.seed,
            "n": len(chosen),
            "source_files": sorted({r["file"] for r in records}),
            "files_represented": len({r["file"] for r in chosen}),
            "videos": chosen,
        }
        Path(a.manifest_out).write_text(json.dumps(manifest, indent=1))
        print(f"selected {len(chosen)} videos from {manifest['files_represented']} files "
              f"-> {a.manifest_out}")
        cf = [c["features"] for c in chosen]
        for key in ["codec", "res", "orientation", "rotation", "fps", "vfr", "start", "duration"]:
            print(f"  {key:12s} {dict(Counter(f.get(key, '-') for f in cf))}")


if __name__ == "__main__":
    main()
