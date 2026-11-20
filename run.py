"""Video indexing baseline (D-015): every frame, batched, single process, one GPU.

    python run.py --input <folder of .parquet files> --output results.jsonl

For every second of every video, the output records which of five objects
(person, car, dog, laptop, cell phone) are present and how many people are visible,
using YOLO11l. The rules are in brief §§3–5. Each design choice is labelled with its
DECISIONS.md entry.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]  # brief §1, output order
PERSON = 0
PREDICT_KW = dict(imgsz=640, conf=0.4, iou=0.7, verbose=False)  # iou must be explicit (D-014)
MAX_BATCH_FRAMES = 32                 # D-015 C2
MAX_BATCH_BYTES = 1 << 30             # D-015 C5: limit by bytes, about 1 GB of BGR frames
HERE = Path(__file__).resolve().parent


class VideoError(Exception):
    """A per-video failure: write one error line and continue with the next video."""


# ---------------------------------------------------------------- output

class Writer:
    """The only writer of the output file (D-015 C1)."""

    def __init__(self, path: str):
        self.f = open(path, "w", buffering=1 << 20)
        self.n_seconds = self.n_errors = 0

    def second(self, video: str, s: int, objects: list[str], people: int):
        self.f.write(json.dumps({"video": video, "second": s, "objects": objects,
                                 "people": people}) + "\n")
        self.n_seconds += 1

    def error(self, video: str, msg: str):
        self.f.write(json.dumps({"video": video, "error": msg[:200]}) + "\n")
        self.n_errors += 1

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


class SecondAccumulator:
    """Groups per-frame results into seconds (brief §5) and writes each second once it's complete.

    A second is complete when a frame from a later second arrives. Seconds with no frames
    in between are written as empty. A frame that belongs to a second that has already
    been written is an error (D-015 C1).
    """

    def __init__(self, video: str, writer: Writer):
        self.video, self.w = video, writer
        self.cur = None          # second currently being collected
        self.next_to_write = 0   # every second below this has been written
        self._reset()

    def _reset(self):
        self.n = 0
        self.obj = [0] * len(TARGETS)
        self.people: list[int] = []

    def add(self, s: int, present: list[bool], n_people: int):
        if s < self.next_to_write or (self.cur is not None and s < self.cur):
            raise VideoError(f"frame timestamp goes back to second {s}")
        if self.cur is None:
            self.cur = s
        elif s > self.cur:
            self._emit_current()
            self.cur = s
        self.n += 1
        for k, p in enumerate(present):
            self.obj[k] += p
        self.people.append(n_people)

    def _emit_current(self):
        for gap in range(self.next_to_write, self.cur):  # seconds with no frames
            self.w.second(self.video, gap, [], 0)
        objects = [TARGETS[k] for k in range(len(TARGETS)) if 2 * self.obj[k] >= self.n]
        people = math.floor(statistics.median(self.people))
        self.w.second(self.video, self.cur, objects, people)
        self.next_to_write = self.cur + 1
        self._reset()

    def finish(self, keep_partial: bool = True):
        """Write the last second being collected. After a failure, keep_partial decides whether it's kept (D-015 #3)."""
        if self.cur is not None and self.n and keep_partial:
            self._emit_current()


# ---------------------------------------------------------------- decoding

def rotation_k(frame, stream) -> int:
    """How many 90° counterclockwise turns np.rot90 needs (D-015 C3).

    frame.rotation is in degrees counterclockwise, from the display matrix. If it's
    missing, the old `rotate` stream tag is used, which is in degrees clockwise.
    """
    ccw = getattr(frame, "rotation", None)
    if not ccw:
        tag = (stream.metadata or {}).get("rotate")
        ccw = -int(float(tag)) if tag not in (None, "") else 0
    return int(round(ccw / 90.0)) % 4


def iter_frames(mp4: bytes):
    """Yield (t_rel: Fraction, bgr ndarray) for every decoded frame, in presentation order.

    t_rel is the time since the first decoded frame (D-015 C4). Raises VideoError
    on any decoding problem. Frames already yielded stay valid.
    """
    import av
    import numpy as np

    try:
        container = av.open(io.BytesIO(mp4), mode="r")
    except Exception as e:  # noqa: BLE001
        raise VideoError(f"could not open video: {type(e).__name__}") from e
    with container:
        if not container.streams.video:
            raise VideoError("no video stream")
        stream = container.streams.video[0]
        t0 = None
        try:
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise VideoError("frame without timestamp")
                tb = frame.time_base or stream.time_base
                t = frame.pts * Fraction(tb.numerator, tb.denominator)
                if t0 is None:
                    t0 = t
                img = frame.to_ndarray(format="bgr24")
                k = rotation_k(frame, stream)
                if k:
                    img = np.ascontiguousarray(np.rot90(img, k=k))
                yield t - t0, img
        except VideoError:
            raise
        except Exception as e:  # noqa: BLE001
            raise VideoError(f"decode failed: {type(e).__name__}: {e}") from e
        if t0 is None:
            raise VideoError("no decodable frames")


# ---------------------------------------------------------------- inference

class Detector:
    def __init__(self, weights: Path):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.model = YOLO(str(weights))
        name_to_id = {v: k for k, v in self.model.names.items()}
        self.target_of = {name_to_id[n]: i for i, n in enumerate(TARGETS)}
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def _predict(self, imgs):
        return self.model.predict(imgs, device=self.device, **PREDICT_KW)

    def detect(self, imgs) -> list[tuple[list[bool], int]]:
        """For each image: (which targets are present, person count). On CUDA out-of-memory, halve the batch and retry (D-015 C5)."""
        try:
            results = self._predict(imgs)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(imgs) == 1:
                raise VideoError("CUDA out of memory on a single frame")
            mid = len(imgs) // 2
            return self.detect(imgs[:mid]) + self.detect(imgs[mid:])
        out = []
        for r in results:
            present = [False] * len(TARGETS)
            n_people = 0
            for c in r.boxes.cls.tolist():
                k = self.target_of.get(int(c))
                if k is not None:
                    present[k] = True
                    n_people += k == PERSON
            out.append((present, n_people))
        return out


# ---------------------------------------------------------------- per video

def process_video(video: str, mp4, det: Detector, w: Writer, keep_partial: bool) -> int:
    acc = SecondAccumulator(video, w)
    batch_t, batch_img, batch_bytes, n_frames = [], [], 0, 0

    def flush_batch():
        nonlocal batch_bytes
        if batch_img:
            for t, (present, n_people) in zip(batch_t, det.detect(batch_img)):
                acc.add(math.floor(t), present, n_people)
        batch_t.clear()
        batch_img.clear()
        batch_bytes = 0

    try:
        if not mp4:
            raise VideoError("empty or missing mp4 bytes")
        try:
            for t, img in iter_frames(mp4):  # batches never mix videos (D-015 C2)
                batch_t.append(t)
                batch_img.append(img)
                batch_bytes += img.nbytes
                n_frames += 1
                if len(batch_img) >= MAX_BATCH_FRAMES or batch_bytes >= MAX_BATCH_BYTES:
                    flush_batch()
        except VideoError:
            flush_batch()  # frames decoded before the failure are still valid
            raise
        flush_batch()
        acc.finish(keep_partial=True)
    except VideoError as e:
        acc.finish(keep_partial=keep_partial)
        w.error(video, str(e))
    except Exception as e:  # noqa: BLE001 - never let one video stop the run
        acc.finish(keep_partial=keep_partial)
        w.error(video, f"unexpected {type(e).__name__}: {e}")
    w.flush()
    return n_frames


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl path")
    ap.add_argument("--weights", default=str(HERE / "weights" / "yolo11l.pt"))
    ap.add_argument("--drop-partial-second", action="store_true",
                    help="after a failure partway through, drop the unfinished second (D-015 #3)")
    ap.add_argument("--only-videos", help=argparse.SUPPRESS)  # dev only: subset manifest JSON
    a = ap.parse_args()

    only = None
    if a.only_videos:  # validation use only: keep row numbers as they are and skip other rows
        only = {v["video"] for v in json.loads(Path(a.only_videos).read_text())["videos"]}

    t_start = time.time()
    files = sorted(Path(a.input).glob("*.parquet"))  # D-015 #1
    det = Detector(Path(a.weights))
    w = Writer(a.output)
    keep_partial = not a.drop_partial_second
    n_videos = n_frames = 0
    file_problems = []

    import pyarrow.parquet as pq

    for path in files:
        try:
            pf = pq.ParquetFile(path)
            n_rows = pf.metadata.num_rows
        except Exception as e:  # noqa: BLE001 - footer unreadable: no valid video IDs (D-015 #2)
            msg = f"{path.name}: unreadable Parquet file ({type(e).__name__}: {e})"
            print(f"WARNING {msg}", file=sys.stderr)
            file_problems.append(msg)
            continue
        row = 0
        try:
            for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                for mp4 in batch.column("mp4").to_pylist():
                    vid = f"{path.name}:{row}"
                    if only is not None and vid not in only:
                        row += 1
                        continue
                    t_v = time.time()
                    nf = process_video(vid, mp4, det, w, keep_partial)
                    n_videos += 1
                    n_frames += nf
                    print(f"{vid} frames={nf} {time.time() - t_v:.1f}s", file=sys.stderr, flush=True)
                    row += 1
        except Exception as e:  # noqa: BLE001 - data unreadable: an error line for each remaining row
            for r in range(row, n_rows):
                w.error(f"{path.name}:{r}", f"could not read row: {type(e).__name__}")
            w.flush()
            msg = f"{path.name}: rows {row}..{n_rows - 1} unreadable ({type(e).__name__}: {e})"
            print(f"WARNING {msg}", file=sys.stderr)
            file_problems.append(msg)

    w.close()
    dt = time.time() - t_start
    print(f"DONE files={len(files)} videos={n_videos} frames={n_frames} "
          f"seconds={w.n_seconds} errors={w.n_errors} file_problems={len(file_problems)} "
          f"wall={dt:.1f}s fps={n_frames / max(dt, 1e-9):.1f}", file=sys.stderr)
    for m in file_problems:
        print(f"INCOMPLETE {m}", file=sys.stderr)
    return 0  # data problems never make the exit code nonzero (D-015 #2)


if __name__ == "__main__":
    sys.exit(main())
