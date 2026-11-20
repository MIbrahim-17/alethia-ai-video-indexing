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
    """About how much memory a buffered YUV 4:2:0 frame holds (0 for a released frame)."""
    if vf is None:
        return 0
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
    def __init__(self, weights: Path, half: bool = False):
        import torch
        from ultralytics import YOLO

        self.half = half  # FP16 inference (D-022); main() uses it by default

        self.torch = torch
        self.model = YOLO(str(weights))
        name_to_id = {v: k for k, v in self.model.names.items()}
        self.target_of = {name_to_id[n]: i for i, n in enumerate(TARGETS)}
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.n_calls = self.n_frames = self.n_oom = 0
        self.batch_cap: dict[tuple, int] = {}  # image shape -> largest batch size known to fit (after an OOM)

    def _release_predictor_state(self):
        """Ultralytics keeps the last batch's images and results on the predictor between calls
        (`batch`, `results`, `dataset`; it rebuilds them on every call). Drop them so a
        finished or failed batch isn't kept in memory."""
        p = getattr(self.model, "predictor", None)
        if p is not None:
            p.batch = p.results = p.dataset = None

    def _predict(self, imgs):
        """One predict call on same-shape images (D-015 C2)."""
        try:
            extra = {"quantize": 16} if self.half else {}  # FP32 calls stay exactly as before
            out = self.model.predict(imgs, device=self.device, **extra, **PREDICT_KW)
        finally:
            self._release_predictor_state()
        self.n_calls += 1
        self.n_frames += len(imgs)
        return out

    def _summarise(self, r) -> tuple[list[bool], int]:
        present = [False] * len(TARGETS)
        n_people = 0
        for c in r.boxes.cls.tolist():
            k = self.target_of.get(int(c))
            if k is not None:
                present[k] = True
                n_people += k == PERSON
        return present, n_people

    @staticmethod
    def _is_oom(e: BaseException) -> bool:
        import torch
        return isinstance(e, torch.cuda.OutOfMemoryError) or (
            isinstance(e, RuntimeError) and "out of memory" in str(e).lower())

    def detect(self, imgs) -> list[tuple[list[bool], int]]:
        """For each image: (which targets are present, person count).

        Images are sent in runs of the same shape, never more than the batch size known to
        fit for that shape. On CUDA out-of-memory (D-015 C5, D-019):
          - the retry happens only after leaving the `except` block, so the exception's
            traceback, which references the failed attempt's GPU tensors, is already gone;
          - the predictor's leftover state is dropped, garbage is collected and the CUDA
            cache is emptied;
          - the batch size for that shape is halved, and stays halved for later batches;
          - an OOM on a single frame is a VideoError for this video only.
        """
        out, i = [], 0
        while i < len(imgs):
            shape = imgs[i].shape
            j = i + 1
            while j < len(imgs) and imgs[j].shape == shape:
                j += 1
            n = min(j - i, self.batch_cap.get(shape, j - i))
            failed = False
            try:
                results = self._predict(imgs[i:i + n])
            except Exception as e:  # noqa: BLE001
                if not self._is_oom(e):
                    raise
                failed = True
            if failed:  # outside the except block: the traceback has been released
                self.n_oom += 1
                self._recover_from_oom()
                if n == 1:
                    raise VideoError("CUDA out of memory on a single frame")
                self.batch_cap[shape] = n // 2
                continue
            out += [self._summarise(r) for r in results]
            del results
            i += n
        return out

    def _recover_from_oom(self):
        import gc

        self._release_predictor_state()
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


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
SECOND_BYTES_CAP = 512 << 20  # a single second holding more than this is run on every frame,
                              # streamed in batches, so it is never buffered whole (D-019)
STATS = {"peak_buffered_bytes": 0, "peak_buffered_frames": 0, "overflow_seconds": 0}


def uniform_positions(n: int, k: int) -> list[int]:
    """k evenly spaced positions in a second of n frames: floor((j + 0.5) * n / k), j = 0..k-1.

    Integer form of the simulator's rule. All positions if n <= k; k = 0 means all.
    """
    if k == 0 or n <= k:
        return list(range(n))
    return sorted({(2 * j + 1) * n // (2 * k) for j in range(k)})


class _Second:
    __slots__ = ("s", "frames", "res", "stage1", "unsure", "overflow")

    def __init__(self, s: int):
        self.s = s
        self.frames = []     # (decoded frame, rotation k) in time order; None once released
        self.res = {}        # position -> (present list, person count) for frames run so far
        self.stage1 = None   # (present tuple, people) from the first stage
        self.unsure = False
        self.overflow = False  # too big to buffer: every frame is run as it arrives


def _held(sec: _Second) -> tuple[int, int]:
    """(bytes, frames) of decoded frames this second still holds."""
    held = [f for f in sec.frames if f is not None]
    return sum(frame_bytes(vf) for vf, _ in held), len(held)


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


INFER_LOG = None   # validation only (--log-inferred): one line per frame sent to the model
_CUR_VIDEO = None


def _run_positions(det, todo):
    """todo: list of (second, position). Runs the model on those frames in bounded batches."""
    if INFER_LOG is not None:
        INFER_LOG.write("".join(f"{_CUR_VIDEO} {sec.s} {p}\n" for sec, p in todo))
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
    """Staged sampling for one video, with one second of look-ahead for the N test.

    Memory (D-019): decoded frames are held only for buffered seconds. At most
    CHUNK_SECONDS + 1 complete seconds or about CHUNK_BYTES are buffered before they are
    processed, and a second's frames and results are dropped as soon as its line is
    written. A second larger than SECOND_BYTES_CAP is run on every frame while it is
    being decoded (in batches), so it never holds more than the cap plus one batch.
    """

    def __init__(self, acc: SecondAccumulator, det, stages, flags):
        self.acc, self.det, self.stages, self.flags = acc, det, stages, flags
        self.pending: list[_Second] = []   # complete seconds not yet written, in order
        self.prev_stage1 = None            # stage-1 result of the last written second
        self.buffered_bytes = self.buffered_frames = 0   # held by self.pending
        self.cur: _Second | None = None    # the second being decoded
        self.cur_bytes = self.cur_frames = 0

    # -- decoding side
    def add_frame(self, s: int, vf, k: int):
        if self.cur is None or s > self.cur.s:
            self.close_current()
            self.cur = _Second(s)
            self.cur_bytes = self.cur_frames = 0
        c = self.cur
        c.frames.append((vf, k))
        self.cur_bytes += frame_bytes(vf)
        self.cur_frames += 1
        self._note_peak()
        if not c.overflow and self.cur_bytes > SECOND_BYTES_CAP:
            c.overflow = True
            STATS["overflow_seconds"] += 1
        if c.overflow:
            self._run_overflow(c, force=False)

    def _run_overflow(self, c: _Second, force: bool):
        """Every frame of an overflow second is run; release each frame once it has been."""
        todo = [(c, p) for p, f in enumerate(c.frames) if f is not None and p not in c.res]
        if todo and (force or len(todo) >= MAX_BATCH_FRAMES):
            _run_positions(self.det, todo)
            for _, p in todo:
                c.frames[p] = None
            self.cur_bytes, self.cur_frames = _held(c)

    def close_current(self):
        if self.cur is None:
            return
        c, self.cur = self.cur, None
        if c.overflow:
            self._run_overflow(c, force=True)
        self.cur_bytes = self.cur_frames = 0
        self.add_second(c)

    def _note_peak(self):
        b = self.buffered_bytes + self.cur_bytes
        f = self.buffered_frames + self.cur_frames
        if b > STATS["peak_buffered_bytes"]:
            STATS["peak_buffered_bytes"] = b
        if f > STATS["peak_buffered_frames"]:
            STATS["peak_buffered_frames"] = f

    # -- inference side
    def add_second(self, sec: _Second):
        if self.pending and sec.s > self.pending[-1].s + 1:  # seconds with no frames
            for g in range(self.pending[-1].s + 1, sec.s):
                self.pending.append(_Second(g))
        elif not self.pending and self.prev_stage1 is None and sec.s > 0:
            for g in range(0, sec.s):
                self.pending.append(_Second(g))
        self.pending.append(sec)
        b, f = _held(sec)
        self.buffered_bytes += b
        self.buffered_frames += f
        self._note_peak()
        if len(self.pending) > CHUNK_SECONDS or self.buffered_bytes >= CHUNK_BYTES:
            self.process(final=False)

    def finish(self):
        self.close_current()
        self.process(final=True)

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
            if sec.stage1 is None:  # from the stage-1 frames only (an overflow second has run all)
                sec.stage1 = _agg([sec.res[p] for p in uniform_positions(len(sec.frames), k1)])
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
        for sec in ready:  # release: written seconds keep no frames and no per-frame results
            sec.frames = []
            sec.res = {}
        del ready
        self.pending = self.pending[-1:] if not final else []
        held = [_held(x) for x in self.pending]
        self.buffered_bytes = sum(h[0] for h in held)
        self.buffered_frames = sum(h[1] for h in held)


def process_video_staged(video: str, mp4, det, w: Writer, keep_partial: bool, stages, flags,
                         frames=None) -> int:
    """Staged sampling for one video. `frames`, if given, replaces iter_raw_frames(mp4) as the
    frame source (the threaded pipeline, D-023); `mp4` is then only checked for emptiness."""
    global _CUR_VIDEO
    _CUR_VIDEO = video
    acc = SecondAccumulator(video, w)
    sv = StagedVideo(acc, det, stages, flags)
    n_frames = 0
    try:
        if not mp4:
            raise VideoError("empty or missing mp4 bytes")
        decode_err = None
        gen = iter_raw_frames(mp4) if frames is None else frames
        while True:
            try:
                t, vf, k = next(gen)
            except StopIteration:
                break
            except VideoError as e:
                decode_err = e  # frames decoded before the failure are still used
                break
            s = math.floor(t)
            if s < 0 or (sv.cur is not None and s < sv.cur.s):
                decode_err = VideoError(f"frame timestamp goes back to second {s}")
                break
            sv.add_frame(s, vf, k)
            del vf
            n_frames += 1
        gen.close()
        sv.finish()
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


# ---------------------------------------------------------------- threaded pipeline (D-023)

QUEUE_BYTES = 512 << 20    # decoded frames waiting in the queue (YUV bytes)
QUEUE_CHUNK_FRAMES = 32    # frames per queue item
STATS["peak_queue_bytes"] = 0


class _Stopped(Exception):
    """The consumer has stopped; the decoder thread exits."""


class _ByteQueue:
    """FIFO limited by the bytes of the frames it holds. One item is always allowed in,
    so a single large chunk can't block forever."""

    def __init__(self, max_bytes: int):
        import collections
        import threading
        self.max_bytes, self.bytes = max_bytes, 0
        self.items = collections.deque()
        self.cv = threading.Condition()
        self.stopped = False

    def put(self, item, nbytes: int = 0):
        with self.cv:
            while self.items and self.bytes + nbytes > self.max_bytes and not self.stopped:
                self.cv.wait()
            if self.stopped:
                raise _Stopped
            self.items.append((item, nbytes))
            self.bytes += nbytes
            STATS["peak_queue_bytes"] = max(STATS["peak_queue_bytes"], self.bytes)
            self.cv.notify_all()

    def get(self):
        with self.cv:
            while not self.items:
                self.cv.wait()
            item, nbytes = self.items.popleft()
            self.bytes -= nbytes
            self.cv.notify_all()
            return item

    def stop(self):
        with self.cv:
            self.stopped = True
            self.items.clear()
            self.bytes = 0
            self.cv.notify_all()


class FrameProducer:
    """The decoder thread: reads Parquet rows in order and decodes each video with
    iter_raw_frames, sending events to the queue in the same order as run_serial works:
      ("file_problem", name, exc)            footer unreadable
      ("rows_unreadable", name, row, n, exc) rows from `row` on can't be read
      ("video", vid, has_bytes)              then, for a video with bytes, ("frames", list)
                                             chunks and exactly one of ("vend",),
                                             ("verr", VideoError) or ("vexc", exception)
      ("done",) at the end, or ("fatal", exc) first if the thread itself failed.
    """

    def __init__(self, files, only):
        import threading
        self.files, self.only = files, only
        self.q = _ByteQueue(QUEUE_BYTES)
        self.thread = threading.Thread(target=self._run, name="decoder", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.q.stop()
        self.thread.join(timeout=10)

    def _run(self):
        try:
            self._produce()
        except _Stopped:
            return
        except BaseException as e:  # noqa: BLE001
            try:
                self.q.put(("fatal", e))
            except _Stopped:
                return
        try:
            self.q.put(("done",))
        except _Stopped:
            pass

    def _produce(self):
        import pyarrow.parquet as pq

        for path in self.files:
            try:
                pf = pq.ParquetFile(path)
                n_rows = pf.metadata.num_rows
            except Exception as e:  # noqa: BLE001
                self.q.put(("file_problem", path.name, e))
                continue
            row = 0
            try:
                for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                    for mp4 in batch.column("mp4").to_pylist():
                        vid = f"{path.name}:{row}"
                        if self.only is None or vid in self.only:
                            self._decode(vid, mp4)
                        row += 1
            except _Stopped:
                raise
            except Exception as e:  # noqa: BLE001 - reading failed: like run_serial
                self.q.put(("rows_unreadable", path.name, row, n_rows, e))

    def _decode(self, vid, mp4):
        self.q.put(("video", vid, bool(mp4)))
        if not mp4:
            return
        chunk, nbytes = [], 0
        try:
            for item in iter_raw_frames(mp4):
                chunk.append(item)
                nbytes += frame_bytes(item[1])
                del item
                if len(chunk) >= QUEUE_CHUNK_FRAMES:
                    self.q.put(("frames", chunk), nbytes)
                    chunk, nbytes = [], 0
        except VideoError as e:
            end = ("verr", e)
        except _Stopped:
            raise
        except Exception as e:  # noqa: BLE001
            end = ("vexc", e)
        else:
            end = ("vend",)
        if chunk:
            self.q.put(("frames", chunk), nbytes)
        self.q.put(end)


class _QueuedFrames:
    """Iterator over one video's frames from the queue; raises the decoder's error after the
    frames decoded before it, like iter_raw_frames. Frames are handed over one at a time and
    not kept, so their release is decided only by process_video_staged (D-019)."""

    def __init__(self, q: _ByteQueue):
        import collections
        self.q, self.ended = q, False
        self.terminal = None  # ("done",) or ("fatal", exc) if the stream ended inside this video
        self.items = collections.deque()

    def __iter__(self):
        return self

    def _next_event(self):
        ev = self.q.get()
        if ev[0] in ("fatal", "done"):  # can only happen if the decoder thread failed
            self.ended, self.terminal = True, ev
            raise RuntimeError("decoder thread failed") from (ev[1] if ev[0] == "fatal" else None)
        return ev

    def __next__(self):
        while not self.items:
            if self.ended:
                raise StopIteration
            ev = self._next_event()
            if ev[0] == "frames":
                self.items.extend(ev[1])
                ev[1].clear()
            elif ev[0] == "vend":
                self.ended = True
            else:  # "verr" or "vexc"
                self.ended = True
                raise ev[1]
            del ev
        return self.items.popleft()

    def close(self):
        """Skip whatever is left of this video (after an early stop, e.g. a timestamp going back)."""
        self.items.clear()
        while not self.ended:
            try:
                ev = self._next_event()
            except RuntimeError:
                return
            if ev[0] in ("vend", "verr", "vexc"):
                self.ended = True
            elif ev[0] == "frames":
                ev[1].clear()


def _log_video(vid, nf, inferred, t_v):
    print(f"{vid} frames={nf} inferred={inferred} {time.time() - t_v:.1f}s", file=sys.stderr, flush=True)


def _file_problem(msg, file_problems):
    print(f"WARNING {msg}", file=sys.stderr)
    file_problems.append(msg)


def run_serial(files, only, det, w, keep_partial, sampling) -> tuple[int, int, list]:
    """The single-threaded loop (D-015, D-017): read, decode and infer in turn. The fallback."""
    import pyarrow.parquet as pq

    n_videos = n_frames = 0
    file_problems = []
    for path in files:
        try:
            pf = pq.ParquetFile(path)
            n_rows = pf.metadata.num_rows
        except Exception as e:  # noqa: BLE001 - footer unreadable: no valid video IDs (D-015 #2)
            _file_problem(f"{path.name}: unreadable Parquet file ({type(e).__name__}: {e})", file_problems)
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
                    if SAMPLING[sampling] is None:
                        nf = process_video(vid, mp4, det, w, keep_partial)
                    else:
                        stages, flags = SAMPLING[sampling]
                        nf = process_video_staged(vid, mp4, det, w, keep_partial, stages, flags)
                    n_videos += 1
                    n_frames += nf
                    _log_video(vid, nf, det.n_frames - inf0, t_v)
                    row += 1
        except Exception as e:  # noqa: BLE001 - data unreadable: an error line for each remaining row
            for r in range(row, n_rows):
                w.error(f"{path.name}:{r}", f"could not read row: {type(e).__name__}")
            w.flush()
            _file_problem(f"{path.name}: rows {row}..{n_rows - 1} unreadable ({type(e).__name__}: {e})",
                          file_problems)
    return n_videos, n_frames, file_problems


def run_threaded(files, only, det, w, keep_partial, sampling) -> tuple[int, int, list]:
    """Decoding in a separate thread, overlapped with inference (D-023). The main thread does
    exactly what run_serial does, in the same order, on the same frames."""
    stages, flags = SAMPLING[sampling]
    prod = FrameProducer(files, only)
    prod.start()
    n_videos = n_frames = 0
    file_problems = []
    try:
        while True:
            ev = prod.q.get()
            kind = ev[0]
            if kind == "done":
                break
            if kind == "fatal":
                raise RuntimeError("decoder thread failed") from ev[1]
            if kind == "file_problem":
                _, name, e = ev
                _file_problem(f"{name}: unreadable Parquet file ({type(e).__name__}: {e})", file_problems)
            elif kind == "rows_unreadable":
                _, name, row, n_rows, e = ev
                for r in range(row, n_rows):
                    w.error(f"{name}:{r}", f"could not read row: {type(e).__name__}")
                w.flush()
                _file_problem(f"{name}: rows {row}..{n_rows - 1} unreadable ({type(e).__name__}: {e})",
                              file_problems)
            elif kind == "video":
                _, vid, has_bytes = ev
                t_v, inf0 = time.time(), det.n_frames
                frames = _QueuedFrames(prod.q) if has_bytes else None
                nf = process_video_staged(vid, has_bytes, det, w, keep_partial, stages, flags,
                                          frames=frames)
                if frames is not None:
                    frames.close()
                    if frames.terminal is not None:  # the decoder thread failed inside this video
                        if frames.terminal[0] == "fatal":
                            raise RuntimeError("decoder thread failed") from frames.terminal[1]
                        break
                n_videos += 1
                n_frames += nf
                _log_video(vid, nf, det.n_frames - inf0, t_v)
            del ev
    finally:
        prod.stop()
    return n_videos, n_frames, file_problems


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
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16",
                    help="model precision; fp16 was promoted in D-022, fp32 is the fallback")
    ap.add_argument("--pipeline", choices=["serial", "threaded"], default="threaded",
                    help="threaded (default, D-023): decode in a separate thread, overlapped with "
                         "inference; serial: the single-threaded fallback")
    ap.add_argument("--only-videos", help=argparse.SUPPRESS)  # dev only: subset manifest JSON
    ap.add_argument("--log-inferred", help=argparse.SUPPRESS)  # dev only: frames sent to the model
    a = ap.parse_args()
    global INFER_LOG
    if a.log_inferred:
        INFER_LOG = open(a.log_inferred, "w", buffering=1 << 20)

    only = None
    if a.only_videos:  # validation use only: keep row numbers as they are and skip other rows
        only = {v["video"] for v in json.loads(Path(a.only_videos).read_text())["videos"]}

    t_start = time.time()
    files = sorted(Path(a.input).glob("*.parquet"))  # D-015 #1
    det = Detector(Path(a.weights), half=a.precision == "fp16")
    w = Writer(a.output)
    keep_partial = not a.drop_partial_second
    pipeline = a.pipeline if SAMPLING[a.sampling] is not None else "serial"  # 'all' is serial only
    runner = run_threaded if pipeline == "threaded" else run_serial
    n_videos, n_frames, file_problems = runner(files, only, det, w, keep_partial, a.sampling)

    w.close()
    if INFER_LOG is not None:
        INFER_LOG.close()
    dt = time.time() - t_start
    print(f"DONE sampling={a.sampling} precision={a.precision} pipeline={pipeline} "
          f"peak_queue_mb={STATS['peak_queue_bytes'] / 2**20:.0f} files={len(files)} videos={n_videos} frames={n_frames} "
          f"inferred={det.n_frames} predict_calls={det.n_calls} oom_retries={det.n_oom} "
          f"peak_buffered_frames={STATS['peak_buffered_frames']} "
          f"peak_buffered_mb={STATS['peak_buffered_bytes'] / 2**20:.0f} "
          f"overflow_seconds={STATS['overflow_seconds']} "
          f"seconds={w.n_seconds} errors={w.n_errors} file_problems={len(file_problems)} "
          f"wall={dt:.1f}s fps={n_frames / max(dt, 1e-9):.1f}", file=sys.stderr)
    for m in file_problems:
        print(f"INCOMPLETE {m}", file=sys.stderr)
    return 0  # data problems never make the exit code nonzero (D-015 #2)


if __name__ == "__main__":
    sys.exit(main())
