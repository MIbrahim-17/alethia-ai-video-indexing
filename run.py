"""Video indexing (D-015 baseline, D-017 staged sampling): batched, single process, one GPU.

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


def frame_to_bgr(vf, k: int):
    """Convert one decoded frame to a BGR array and apply its rotation (D-015 C3)."""
    import numpy as np

    img = vf.to_ndarray(format="bgr24")
    if k:
        img = np.ascontiguousarray(np.rot90(img, k=k))
    return img


def frame_bytes(vf) -> int:
    """About how much memory a buffered YUV 4:2:0 frame holds."""
    try:
        return vf.width * vf.height * 3 // 2
    except AttributeError:
        return 1


def iter_raw_frames(mp4: bytes):
    """Yield (t_rel: Fraction, decoded frame, rotation k) for every frame, in presentation order.

    t_rel is the time since the first decoded frame (D-015 C4). The frame isn't converted
    yet, so frames that are never sent to the model cost only the decode. Raises
    VideoError on any decoding problem. Frames already yielded stay valid.
    """
    import av

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
                yield t - t0, frame, rotation_k(frame, stream)
        except VideoError:
            raise
        except Exception as e:  # noqa: BLE001
            raise VideoError(f"decode failed: {type(e).__name__}: {e}") from e
        if t0 is None:
            raise VideoError("no decodable frames")


def iter_frames(mp4: bytes):
    """Yield (t_rel: Fraction, bgr ndarray) for every decoded frame (the every-frame path)."""
    for t, vf, k in iter_raw_frames(mp4):
        yield t, frame_to_bgr(vf, k)


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

        self.n_calls = self.n_frames = 0

    def _predict(self, imgs):
        """Run predict on runs of same-shape images only, so Ultralytics's `auto` letterbox
        is the same as for single frames (D-015 C2). Real videos never change shape."""
        out, i = [], 0
        while i < len(imgs):
            j = i + 1
            while j < len(imgs) and imgs[j].shape == imgs[i].shape:
                j += 1
            out += self.model.predict(imgs[i:j], device=self.device, **PREDICT_KW)
            self.n_calls += 1
            self.n_frames += j - i
            i = j
        return out

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


# ---------------------------------------------------------------- staged sampling (D-016, D-017)

# name -> (stages, uncertainty flags). A stage of 0 means every frame of the second.
# These are the exact rules of validation/simulate_sampling.py strat_staged().
SAMPLING = {
    "staged-1>3>all-VPN": ((1, 3, 0), "VPN"),  # provisional default (D-017)
    "all": None,                               # every-frame baseline (D-015), kept as fallback
}
DEFAULT_SAMPLING = "staged-1>3>all-VPN"
CHUNK_SECONDS = 32           # seconds buffered before running the model on them
CHUNK_BYTES = 1 << 30        # and at most about 1 GB of buffered YUV frames (D-015 C5)


def uniform_positions(n: int, k: int) -> list[int]:
    """k evenly spaced positions in a second of n frames: floor((j + 0.5) * n / k), j = 0..k-1.

    Integer form of the simulator's rule. All positions if n <= k; k = 0 means all.
    """
    if k == 0 or n <= k:
        return list(range(n))
    return sorted({(2 * j + 1) * n // (2 * k) for j in range(k)})


class _Second:
    __slots__ = ("s", "frames", "res", "stage1", "unsure")

    def __init__(self, s: int):
        self.s = s
        self.frames = []     # (decoded frame, rotation k), in time order
        self.res = {}        # position -> (present list, person count) for frames run so far
        self.stage1 = None   # (present tuple, people) from the first stage
        self.unsure = False


def _agg(results) -> tuple[tuple[bool, ...], int]:
    """brief §5 over a list of per-frame results; an empty list gives (no objects, 0)."""
    if not results:
        return (False,) * len(TARGETS), 0
    n = len(results)
    present = tuple(2 * sum(r[0][k] for r in results) >= n for k in range(len(TARGETS)))
    return present, math.floor(statistics.median(r[1] for r in results))


def _unsure_vp(sec: _Second, flags: str) -> bool:
    """V: the frames run so far disagree on some object. P: they disagree on the person count."""
    rs = list(sec.res.values())
    if "V" in flags and any(len({r[0][k] for r in rs}) > 1 for k in range(len(TARGETS))):
        return True
    if "P" in flags and len({r[1] for r in rs}) > 1:
        return True
    return False


def _run_positions(det, todo):
    """todo: list of (second, position). Runs the model on those frames in bounded batches."""
    i = 0
    while i < len(todo):
        batch, nbytes = [], 0
        while i < len(todo) and len(batch) < MAX_BATCH_FRAMES and nbytes < MAX_BATCH_BYTES:
            sec, pos = todo[i]
            vf, k = sec.frames[pos]
            img = frame_to_bgr(vf, k)
            batch.append((sec, pos, img))
            nbytes += img.nbytes
            i += 1
        for (sec, pos, _), r in zip(batch, det.detect([b[2] for b in batch])):
            sec.res[pos] = r


class StagedVideo:
    """Staged sampling for one video, with one second of look-ahead for the N test."""

    def __init__(self, acc: SecondAccumulator, det, stages, flags):
        self.acc, self.det, self.stages, self.flags = acc, det, stages, flags
        self.pending: list[_Second] = []   # complete seconds not yet written, in order
        self.prev_stage1 = None            # stage-1 result of the last written second
        self.buffered_bytes = 0

    def add_second(self, sec: _Second):
        if self.pending and sec.s > self.pending[-1].s + 1:  # seconds with no frames
            for g in range(self.pending[-1].s + 1, sec.s):
                self.pending.append(_Second(g))
        elif not self.pending and self.prev_stage1 is None and sec.s > 0:
            for g in range(0, sec.s):
                self.pending.append(_Second(g))
        self.pending.append(sec)
        self.buffered_bytes += sum(frame_bytes(vf) for vf, _ in sec.frames)
        if len(self.pending) > CHUNK_SECONDS or self.buffered_bytes >= CHUNK_BYTES:
            self.process(final=False)

    def process(self, final: bool):
        """Run the stages on the buffered seconds and write all whose neighbours are known."""
        if not self.pending:
            return
        k1 = self.stages[0]
        # Stage 1 for every buffered second that hasn't had it.
        todo = []
        for sec in self.pending:
            if sec.stage1 is None:
                todo += [(sec, p) for p in uniform_positions(len(sec.frames), k1) if p not in sec.res]
        _run_positions(self.det, todo)
        for sec in self.pending:
            if sec.stage1 is None:
                sec.stage1 = _agg(list(sec.res.values()))
        # The last second waits for its next neighbour unless the video has ended.
        ready = self.pending if final else self.pending[:-1]
        for i, sec in enumerate(ready):
            n = len(sec.frames)
            sec.unsure = False
            if len(sec.res) < n:
                sec.unsure = _unsure_vp(sec, self.flags)
                if "N" in self.flags:
                    prev = self.pending[i - 1].stage1 if i > 0 else self.prev_stage1
                    nxt = self.pending[i + 1].stage1 if i + 1 < len(self.pending) else None
                    if any(o is not None and o != sec.stage1 for o in (prev, nxt)):
                        sec.unsure = True
        # Later stages, each batched over all seconds still uncertain.
        later_flags = self.flags.replace("N", "")
        for k in self.stages[1:]:
            todo = []
            for sec in ready:
                if sec.unsure:
                    todo += [(sec, p) for p in uniform_positions(len(sec.frames), k) if p not in sec.res]
            if not todo and not any(sec.unsure for sec in ready):
                break
            _run_positions(self.det, todo)
            for sec in ready:
                if sec.unsure:
                    sec.unsure = len(sec.res) < len(sec.frames) and _unsure_vp(sec, later_flags)
        # Write: feed each second's run frames, in time order, to the accumulator.
        for sec in ready:
            for p in sorted(sec.res):
                present, n_people = sec.res[p]
                self.acc.add(sec.s, present, n_people)
        if ready:
            self.prev_stage1 = ready[-1].stage1
        self.pending = self.pending[len(ready):]
        self.buffered_bytes = sum(frame_bytes(vf) for sec in self.pending for vf, _ in sec.frames)
        for sec in ready:
            sec.frames = []


def process_video_staged(video: str, mp4, det, w: Writer, keep_partial: bool, stages, flags) -> int:
    acc = SecondAccumulator(video, w)
    sv = StagedVideo(acc, det, stages, flags)
    n_frames = 0
    try:
        if not mp4:
            raise VideoError("empty or missing mp4 bytes")
        cur, decode_err = None, None
        gen = iter_raw_frames(mp4)
        while True:
            try:
                t, vf, k = next(gen)
            except StopIteration:
                break
            except VideoError as e:
                decode_err = e  # frames decoded before the failure are still used
                break
            s = math.floor(t)
            if s < 0 or (cur is not None and s < cur.s):
                decode_err = VideoError(f"frame timestamp goes back to second {s}")
                break
            if cur is None or s > cur.s:
                if cur is not None:
                    sv.add_second(cur)
                cur = _Second(s)
            cur.frames.append((vf, k))
            n_frames += 1
        if cur is not None:
            sv.add_second(cur)
        sv.process(final=True)
        if decode_err is not None:
            raise decode_err
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
    ap.add_argument("--sampling", choices=list(SAMPLING), default=DEFAULT_SAMPLING,
                    help="which frames go through the model (D-017); 'all' is the every-frame fallback")
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
                    t_v, inf0 = time.time(), det.n_frames
                    if SAMPLING[a.sampling] is None:
                        nf = process_video(vid, mp4, det, w, keep_partial)
                    else:
                        stages, flags = SAMPLING[a.sampling]
                        nf = process_video_staged(vid, mp4, det, w, keep_partial, stages, flags)
                    n_videos += 1
                    n_frames += nf
                    print(f"{vid} frames={nf} inferred={det.n_frames - inf0} {time.time() - t_v:.1f}s",
                          file=sys.stderr, flush=True)
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
    print(f"DONE sampling={a.sampling} files={len(files)} videos={n_videos} frames={n_frames} "
          f"inferred={det.n_frames} predict_calls={det.n_calls} "
          f"seconds={w.n_seconds} errors={w.n_errors} file_problems={len(file_problems)} "
          f"wall={dt:.1f}s fps={n_frames / max(dt, 1e-9):.1f}", file=sys.stderr)
    for m in file_problems:
        print(f"INCOMPLETE {m}", file=sys.stderr)
    return 0  # data problems never make the exit code nonzero (D-015 #2)


if __name__ == "__main__":
    sys.exit(main())
