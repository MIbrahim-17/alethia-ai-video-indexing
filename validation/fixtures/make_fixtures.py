"""Generate the D-013 Critical fixture suite as one Parquet file plus a manifest.

Each fixture is a row with an `mp4` column (and an empty `json` column, as in FineVideo).
The manifest records, for each row, how the expected output is derived:
  - "oracle": expected lines come from an independent path. cv2.VideoCapture decodes
    (with its own automatic rotation), the timestamps are the ones we intended,
    stock single-frame predict runs on each frame, and reference_aggregate groups them.
  - "error_only": exactly one error line and no second lines.
  - "observe": no fixed expectation yet. The behaviour is reported for a decision.

Frame content: P is a real FineVideo frame where YOLO11l is confident there is a
person, and B is a flat grey frame. Alternating P and B at chosen timestamps means a
frame put in the wrong second changes the expected output, so mistakes in second
assignment show up.

Usage (on the server):
  python validation/fixtures/make_fixtures.py --out work/fixtures
"""

from __future__ import annotations

import argparse
import io
import json
import random
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

W, H = 640, 360
TB = Fraction(1, 90000)  # the time base used for all fixtures


def person_frame(src_parquet: str, row: int, frame_idx: int) -> np.ndarray:
    pf = pq.ParquetFile(src_parquet)
    for r, b in enumerate(pf.iter_batches(batch_size=1, columns=["mp4"])):
        if r == row:
            mp4 = b.column("mp4")[0].as_py()
            break
    with av.open(io.BytesIO(mp4)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i == frame_idx:
                img = f.to_ndarray(format="bgr24")
                break
    return np.ascontiguousarray(img[:H, :W])


def encode(frames, times, codec="libx264", rotation=None, faststart=False, size=(W, H),
           x264_opts=None) -> bytes:
    """Encode frames with exact presentation times (in seconds, as Fractions). Returns the mp4 bytes."""
    import tempfile

    # faststart rewrites the file to move moov to the front, which needs a real file.
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    opts = {"movflags": "+faststart"} if faststart else {}
    with av.open(tmp.name, "w", format="mp4", options=opts) as c:
        s = c.add_stream(codec, rate=30)
        s.width, s.height = size
        s.pix_fmt = "yuv420p"
        s.time_base = TB
        s.codec_context.time_base = TB
        if codec == "libx264":
            s.options = x264_opts or {"crf": "0", "bf": "0"}  # default: near-lossless, no B-frames
        if rotation is not None:
            s.set_display_rotation(rotation)
        for img, t in zip(frames, times):
            vf = av.VideoFrame.from_ndarray(img, format="bgr24")
            vf.pts = int(t / TB)
            vf.time_base = TB
            for pkt in s.encode(vf):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink()
    return data


def audio_only() -> bytes:
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        s = c.add_stream("aac", rate=44100)
        s.layout = "mono"
        for i in range(20):
            af = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp",
                                            layout="mono")
            af.sample_rate = 44100
            af.pts = i * 1024
            for pkt in s.encode(af):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return buf.getvalue()


def F(x) -> Fraction:
    return Fraction(x).limit_denominator(90000)


def build(P: np.ndarray):
    B = np.full_like(P, 128)
    eps = Fraction(1, 90000)  # one tick of the time base
    fx = []  # (name, mp4 bytes or None, expectation, frame times or None)

    def add(name, data, kind, times=None, note=""):
        fx.append({"name": name, "data": data, "kind": kind,
                   "times": [str(t) for t in times] if times else None, "note": note})

    # Rotation 90/180/270: P only, 2 s at 10 fps. The person's orientation decides detection.
    t10 = [F(i / 10) for i in range(20)]
    for rot in (90, 180, 270):
        add(f"rotation_{rot}", encode([P] * 20, t10, rotation=rot), "oracle", t10,
            "displayed orientation must match cv2 automatic rotation, applied exactly once")

    # Variable frame rate. The P/B pattern makes a wrong second visible.
    vfr_t = [F(0), F(0.05), F(0.4), F(0.41), F(0.97), F(1.3), F(1.31), F(1.32), F(2.9), F(3.05)]
    vfr_f = [B, P, P, B, B, P, B, B, P, P]
    add("vfr", encode(vfr_f, vfr_t), "oracle", vfr_t)

    # Video starting at a nonzero timestamp (5.3 s): the first frame must be second 0.
    nz_t = [F(5.3 + i / 10) for i in range(25)]
    add("nonzero_start", encode([P if i % 3 else B for i in range(25)], nz_t), "oracle", nz_t)

    # Whole-second boundaries: frames one tick before, exactly on, and one tick after.
    bd_t = [F(0), 1 - eps, F(1), 1 + eps, F(1.5), 2 - eps, F(2), F(2.5)]
    bd_f = [B, P, B, B, B, P, P, B]
    # Expected: sec0 = {B,P} -> person present (tie). sec1 = {B,B,B,P} -> absent.
    # sec2 = {P,B} -> present. Moving any boundary frame flips one of these.
    add("second_boundaries", encode(bd_f, bd_t), "oracle", bd_t)

    # A gap in the timestamps: frames at 0.2 and 0.5, then 3.4 → seconds 1 and 2 are empty.
    gap_t = [F(0.2), F(0.5), F(3.4), F(3.6)]
    add("timestamp_gap", encode([P, P, P, B], gap_t), "oracle", gap_t)

    # A single frame, and a sub-second video.
    add("one_frame", encode([P], [F(0)]), "oracle", [F(0)])
    sub_t = [F(i / 10) for i in range(4)]
    add("sub_second", encode([P, B, P, P], sub_t), "oracle", sub_t)

    # Damaged inputs.
    good = encode([P] * 30, [F(i / 10) for i in range(30)])
    # Damaged-video sources must be realistic: a lossy encode, a keyframe every 10 frames,
    # and frames of similar size. Otherwise one huge keyframe holds most of the file and
    # any damage hits frame 0 (first fixture version, 2026-11-20).
    rng = np.random.default_rng(0)
    var = [np.clip(P.astype(int) + rng.integers(-20, 20, P.shape), 0, 255).astype(np.uint8)
           for _ in range(100)]
    t100 = [F(i / 10) for i in range(100)]
    real_opts = {"crf": "23", "g": "10", "bf": "0"}
    good_fs = encode(var, t100, faststart=True, x264_opts=real_opts)
    add("empty_bytes", b"", "error_only")
    add("null_mp4", None, "error_only")
    add("non_video_bytes", random.Random(0).randbytes(50_000), "error_only")
    add("valid_after_garbage", good, "oracle", [F(i / 10) for i in range(30)],
        "must be processed normally after the garbage row")
    add("audio_only", audio_only(), "error_only")
    add("truncated_moov_at_end", good[: len(good) // 2], "error_only",
        note="moov atom cut off, so the container can't be opened")
    # Expected result: the frames cv2 can decode, using their intended times, plus exactly one error line.
    add("truncated_faststart", good_fs[: int(len(good_fs) * 0.6)], "oracle_partial", t100,
        note="moov at the front, mdat cut at 60%: keep the decoded seconds, then one error line")
    bad = bytearray(good_fs)
    mid = len(bad) // 2
    bad[mid: mid + 4000] = random.Random(1).randbytes(4000)
    # The FFmpeg H.264 decoder hides this damage (cv2 and PyAV both decode all 100 frames).
    add("corrupted_middle", bytes(bad), "oracle", t100,
        note="4 kB of random bytes in mdat, hidden by the decoder; must match cv2")
    return fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="data/finevideo/data/train-00000-of-01357.parquet")
    ap.add_argument("--src-row", type=int, default=1)
    ap.add_argument("--src-frame", type=int, default=0, help="a frame with a confident person")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "input").mkdir(parents=True, exist_ok=True)
    P = person_frame(a.src, a.src_row, a.src_frame)
    np.save(out / "P.npy", P)
    fx = build(P)
    name = "fixtures-critical.parquet"
    table = pa.table({"mp4": pa.array([f["data"] for f in fx], pa.binary()),
                      "json": pa.array(["{}"] * len(fx), pa.string())})
    pq.write_table(table, out / "input" / name)
    manifest = [{"row": i, "video": f"{name}:{i}", **{k: v for k, v in f.items() if k != "data"}}
                for i, f in enumerate(fx)]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(fx)} fixtures -> {out / 'input' / name}")


if __name__ == "__main__":
    main()
