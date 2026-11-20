"""Unit tests for run.py's aggregation and failure handling (validation layer 3, D-013).

No GPU or video is needed: the decoder and detector are replaced with fakes, so each
rule can be checked exactly. Run with:  python validation/test_run_unit.py
"""

from __future__ import annotations

import io
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run  # noqa: E402

P, C, D, L, PH = range(5)  # person, car, dog, laptop, cell phone


class FakeWriter(run.Writer):
    def __init__(self):
        self.f = io.StringIO()
        self.n_seconds = self.n_errors = 0

    def lines(self):
        return [json.loads(x) for x in self.f.getvalue().splitlines()]


class FakeDetector:
    """Each 'image' is a list of target indices (one entry per detected box)."""

    def detect(self, imgs):
        out = []
        for boxes in imgs:
            present = [k in boxes for k in range(5)]
            out.append((present, boxes.count(P)))
        return out


def frames(spec, fail_after=None):
    """spec: a list of (time in seconds, boxes). Raises VideoError after `fail_after` frames."""
    def gen(_mp4):
        for i, (t, boxes) in enumerate(spec):
            if fail_after is not None and i == fail_after:
                raise run.VideoError("decode failed: simulated")
            yield Fraction(t), _Img(boxes)
    return gen


class _Img(list):
    nbytes = 1


def run_video(spec, fail_after=None, keep_partial=True):
    w = FakeWriter()
    orig = run.iter_frames
    run.iter_frames = frames(spec, fail_after)
    try:
        run.process_video("v:0", b"x", FakeDetector(), w, keep_partial)
    finally:
        run.iter_frames = orig
    return w.lines()


def secs(lines):
    return [(r["second"], r["objects"], r["people"]) for r in lines if "second" in r]


def errs(lines):
    return [r for r in lines if "error" in r]


def test_tie_counts_as_present_and_median_floor():
    # 4 frames: person counts 0, 0, 1, 1. Present in 2 of 4 frames, so present (tie).
    # Median 0.5 rounds down to 0.
    lines = run_video([(0.0, []), (0.25, []), (0.5, [P]), (0.75, [P])])
    assert secs(lines) == [(0, ["person"], 0)], lines


def test_median_halfway_rounds_down():
    lines = run_video([(0.0, [P, P]), (0.5, [P, P, P])])  # median 2.5 -> 2
    assert secs(lines) == [(0, ["person"], 2)], lines


def test_under_half_is_absent_and_object_order():
    lines = run_video([(0.0, [PH, C]), (0.3, [C]), (0.6, [])])  # car 2/3 present, phone 1/3 absent
    assert secs(lines) == [(0, ["car"], 0)], lines
    lines = run_video([(0.0, [PH, D, P, L, C])])
    assert secs(lines)[0][1] == ["person", "car", "dog", "laptop", "cell phone"]


def test_gap_seconds_are_empty_lines():
    lines = run_video([(0.1, [P]), (3.2, [P])])
    assert secs(lines) == [(0, ["person"], 1), (1, [], 0), (2, [], 0), (3, ["person"], 1)], lines


def test_boundaries_are_exact():
    tick = Fraction(1, 90000)
    lines = run_video([(0, []), (1 - tick, [P]), (1, []), (1 + tick, []), (2 - tick, [P]), (2, [P])])
    assert [s for s, *_ in secs(lines)] == [0, 1, 2]
    assert secs(lines) == [(0, ["person"], 0), (1, [], 0), (2, ["person"], 1)], lines


def test_last_partial_second_written():
    lines = run_video([(0.0, [P]), (1.0, [P]), (1.1, [P])])
    assert secs(lines)[-1][0] == 1 and not errs(lines)


def test_partial_failure_keeps_lines_and_partial_second():
    spec = [(i / 10, [P]) for i in range(35)]  # 3.5 s of frames
    lines = run_video(spec, fail_after=25)     # fails at t = 2.5 s
    assert [s for s, *_ in secs(lines)] == [0, 1, 2], lines  # second 2 is partial but kept
    assert len(errs(lines)) == 1 and lines[-1]["error"].startswith("decode failed")


def test_partial_failure_drop_mode():
    spec = [(i / 10, [P]) for i in range(35)]
    lines = run_video(spec, fail_after=25, keep_partial=False)
    assert [s for s, *_ in secs(lines)] == [0, 1], lines
    assert len(errs(lines)) == 1


def test_timestamp_going_back_is_error():
    lines = run_video([(0.0, [P]), (1.2, [P]), (0.5, [P])])
    assert [s for s, *_ in secs(lines)] == [0, 1] and len(errs(lines)) == 1, lines


def test_failure_on_first_frame_gives_error_only():
    lines = run_video([(0.0, [P])], fail_after=0)
    assert secs(lines) == [] and len(errs(lines)) == 1


def test_empty_bytes_error_only():
    w = FakeWriter()
    run.process_video("v:0", b"", FakeDetector(), w, True)
    assert w.lines() == [{"video": "v:0", "error": "empty or missing mp4 bytes"}]
    w = FakeWriter()
    run.process_video("v:1", None, FakeDetector(), w, True)
    assert len(w.lines()) == 1 and "error" in w.lines()[0]


def test_rotation_k_mapping():
    class Fr:
        def __init__(self, r): self.rotation = r

    class St:
        def __init__(self, tag=None): self.metadata = {"rotate": tag} if tag is not None else {}

    assert [run.rotation_k(Fr(r), St()) for r in (0, 90, -90, 180, -180)] == [0, 1, 3, 2, 2]
    assert run.rotation_k(Fr(0), St("90")) == 3  # the old clockwise tag: 90° cw = 270° ccw


# ---------------------------------------------------------------- staged sampling (D-017)

class CountingDetector(FakeDetector):
    def __init__(self):
        self.n_frames = self.n_calls = 0
        self.seen = []

    def detect(self, imgs):
        self.n_calls += 1
        self.n_frames += len(imgs)
        self.seen += [im.i for im in imgs]
        return super().detect(imgs)


class _RawImg(list):
    nbytes = 1

    def __init__(self, boxes, i):
        super().__init__(boxes)
        self.i = i


def run_staged(spec, fail_after=None, keep_partial=True, stages=(1, 3, 0), flags="VPN"):
    w, det = FakeWriter(), CountingDetector()
    orig = run.iter_raw_frames, run.frame_to_bgr

    def gen(_mp4):
        for i, (t, boxes) in enumerate(spec):
            if fail_after is not None and i == fail_after:
                raise run.VideoError("decode failed: simulated")
            yield Fraction(t), _RawImg(boxes, i), 0

    run.iter_raw_frames, run.frame_to_bgr = gen, (lambda vf, k: vf)
    try:
        run.process_video_staged("v:0", b"x", det, w, keep_partial, stages, flags)
    finally:
        run.iter_raw_frames, run.frame_to_bgr = orig
    return w.lines(), det


def _random_video(rng, n_sec):
    """Random frame rate per video, random gaps, objects that persist for a few seconds."""
    import numpy as np  # noqa: F401
    fps = int(rng.choice([1, 2, 3, 5, 12, 24, 25, 30, 60]))
    spec, state = [], [0, 0, 0, 0, 0]
    t = Fraction(0)
    for s in range(n_sec):
        if s > 0 and rng.random() < 0.05:
            continue  # a second with no frames
        state = [x if rng.random() < 0.8 else int(rng.integers(0, 3)) for x in state]
        for j in range(fps):
            boxes = []
            for k, x in enumerate(state):
                c = x + (1 if rng.random() < 0.15 else 0) - (1 if x and rng.random() < 0.15 else 0)
                boxes += [k] * int(c)
            spec.append((Fraction(s) + Fraction(j, fps), boxes))
    return spec


def _simulator_output(spec, stages, flags):
    import numpy as np
    sys.path.insert(0, str(ROOT / "validation"))
    import simulate_sampling as sim

    sec = np.array([int(t // 1) for t, _ in spec])
    pres = np.array([[k in b for k in range(5)] for _, b in spec], bool)
    cnt = np.array([b.count(P) for _, b in spec])
    n_sec = int(sec.max()) + 1
    v = {"sec": sec, "t": np.array([float(t) for t, _ in spec]), "pres": pres, "cnt": cnt,
         "minconf": np.ones(len(spec)), "bounds": np.searchsorted(sec, np.arange(n_sec + 1)),
         "n_sec": n_sec}
    sel = sim.strat_staged(v, stages, flags)
    out = []
    for s, idx in enumerate(sel):
        p, c = sim.agg(v, idx)
        out.append((s, [run.TARGETS[k] for k in range(5) if p[k]], c))
    return out, sorted(int(i) for idx in sel for i in idx)


def test_staged_matches_simulator_exactly():
    import numpy as np
    rng = np.random.default_rng(7)
    for trial in range(60):
        spec = _random_video(rng, int(rng.integers(1, 120)))  # up to ~4 chunks of 32 seconds
        lines, det = run_staged(spec)
        want, want_frames = _simulator_output(spec, (1, 3, 0), "VPN")
        assert secs(lines) == want, f"trial {trial}: output differs from the simulator"
        assert sorted(det.seen) == want_frames, f"trial {trial}: different frames sent to the model"
        assert len(set(det.seen)) == len(det.seen), f"trial {trial}: a frame was run twice"


def test_staged_all_frames_equals_baseline():
    import numpy as np
    rng = np.random.default_rng(3)
    for _ in range(20):
        spec = _random_video(rng, int(rng.integers(1, 80)))
        lines, det = run_staged(spec, stages=(0,), flags="")
        assert secs(lines) == secs(run_video(spec)) and det.n_frames == len(spec)


def test_staged_partial_failure_keeps_lines():
    spec = [(i / 10, [P]) for i in range(35)]
    lines, _ = run_staged(spec, fail_after=25)
    assert [s for s, *_ in secs(lines)] == [0, 1, 2] and len(errs(lines)) == 1, lines
    lines, _ = run_staged(spec, fail_after=25, keep_partial=False)
    assert [s for s, *_ in secs(lines)] == [0, 1] and len(errs(lines)) == 1, lines


def test_staged_timestamp_going_back_is_error():
    lines, _ = run_staged([(0.0, [P]), (1.2, [P]), (0.5, [P])])
    assert [s for s, *_ in secs(lines)] == [0, 1] and len(errs(lines)) == 1, lines


def test_staged_first_frame_failure_and_empty():
    lines, _ = run_staged([(0.0, [P])], fail_after=0)
    assert secs(lines) == [] and len(errs(lines)) == 1
    w = FakeWriter()
    run.process_video_staged("v:0", b"", CountingDetector(), w, True, (1, 3, 0), "VPN")
    assert w.lines() == [{"video": "v:0", "error": "empty or missing mp4 bytes"}]


# ---------------------------------------------------------------- memory (D-019)

class _BigImg(_RawImg):
    """A fake frame that reports a 4K size, so the byte accounting sees real-sized frames."""
    width, height = 3840, 2160


def _run_tracked(spec, frame_cls=_BigImg):
    """Run the staged path and check, after every process() call, that every frame of every
    second already written is no longer referenced anywhere (weak references are dead)."""
    import gc
    import weakref
    refs = {}  # second -> weak references to its frames
    violations = []
    w, det = FakeWriter(), CountingDetector()
    orig = run.iter_raw_frames, run.frame_to_bgr, run.StagedVideo.process

    def gen(_mp4):
        for i, (t, boxes) in enumerate(spec):
            f = frame_cls(boxes, i)
            refs.setdefault(int(t // 1), []).append(weakref.ref(f))
            yield Fraction(t), f, 0

    def tracked_process(self, final):
        orig[2](self, final)
        alive_from = min([x.s for x in self.pending] + ([self.cur.s] if self.cur else []) + [10**9])
        for s, rs in refs.items():
            if s < alive_from and any(r() is not None for r in rs):
                violations.append(s)

    run.iter_raw_frames, run.frame_to_bgr, run.StagedVideo.process = gen, (lambda vf, k: vf), tracked_process
    for k in run.STATS:
        run.STATS[k] = 0
    try:
        run.process_video_staged("v:0", b"x", det, w, True, (1, 3, 0), "VPN")
    finally:
        run.iter_raw_frames, run.frame_to_bgr, run.StagedVideo.process = orig
    gc.collect()
    leftover = sum(r() is not None for rs in refs.values() for r in rs)
    return w.lines(), det, sorted(set(violations)), leftover


def test_frames_released_once_second_written():
    import numpy as np
    rng = np.random.default_rng(11)
    for _ in range(10):
        spec = _random_video(rng, int(rng.integers(40, 120)))
        lines, _, violations, leftover = _run_tracked(spec, _RawImg)
        assert violations == [], f"frames of written seconds still referenced: {violations[:5]}"
        assert leftover == 0, f"{leftover} frames still referenced after the video"


def test_buffer_bounded_for_4k_frames():
    # 90 s of 4K at 30 fps: about 33 GB of YUV if it were all buffered.
    spec = [(Fraction(i, 30), [P] if (i // 30) % 7 else []) for i in range(90 * 30)]
    lines, _, violations, leftover = _run_tracked(spec)
    fb = 3840 * 2160 * 3 // 2
    bound = run.CHUNK_BYTES + run.SECOND_BYTES_CAP + fb
    assert run.STATS["peak_buffered_bytes"] <= bound, run.STATS
    assert violations == [] and leftover == 0
    assert [s for s, *_ in secs(lines)] == list(range(90))
    want, _ = _simulator_output(spec, (1, 3, 0), "VPN")
    assert secs(lines) == want  # bounding the buffer doesn't change any result


def test_oversized_second_is_streamed_not_buffered():
    # Second 1 holds 3,000 4K frames (about 37 GB of YUV); seconds 0 and 2..5 are normal.
    spec = [(Fraction(i, 30), [P]) for i in range(30)]
    spec += [(1 + Fraction(i, 3000), [P] * (1 + i % 3)) for i in range(3000)]
    spec += [(2 + Fraction(i, 30), [P, P]) for i in range(4 * 30)]
    lines, det, violations, leftover = _run_tracked(spec)
    fb = 3840 * 2160 * 3 // 2
    max_frames = run.SECOND_BYTES_CAP // fb + run.MAX_BATCH_FRAMES + 2
    assert run.STATS["overflow_seconds"] == 1, run.STATS
    assert run.STATS["peak_buffered_frames"] <= 30 + max_frames, run.STATS
    assert violations == [] and leftover == 0
    # The oversized second is computed from every frame, so it equals the reference rule;
    # the other seconds equal the simulator.
    want, _ = _simulator_output(spec, (1, 3, 0), "VPN")
    got = secs(lines)
    full = run_video(spec)
    assert got[1] == secs(full)[1], (got[1], secs(full)[1])
    assert [g for g in got if g[0] != 1] == [x for x in want if x[0] != 1]


# ---------------------------------------------------------------- threaded pipeline (D-023)

class _Boom(BaseException):
    """Not an Exception: escapes every per-video handler, like a crash in the decoder thread."""


def _pipeline_inputs(tmp):
    """Two Parquet files plus an unreadable one. Each mp4 value is a key into SPECS."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(5)
    specs = {}
    rows = []
    for i in range(14):
        key = f"v{i}".encode()
        spec = _random_video(rng, int(rng.integers(1, 70)))
        mode = {3: "fail", 6: "back", 9: "exc"}.get(i)
        specs[key] = (spec, mode)
        rows.append(key)
    rows[4], rows[11] = b"", None  # empty and missing bytes
    for name, part in (("a.parquet", rows[:7]), ("b.parquet", rows[7:])):
        pq.write_table(pa.table({"mp4": pa.array(part, pa.binary())}), tmp / name)
    (tmp / "c.parquet").write_bytes(b"not a parquet file")
    return specs


def _run_pipeline(kind, tmp, specs, crash_key=None):
    w, det = FakeWriter(), CountingDetector()
    orig = run.iter_raw_frames, run.frame_to_bgr

    def gen(mp4):
        spec, mode = specs[mp4]
        if mp4 == crash_key:
            raise _Boom()
        for i, (t, boxes) in enumerate(spec):
            if mode == "fail" and i == len(spec) // 2:
                raise run.VideoError("decode failed: simulated")
            if mode == "exc" and i == len(spec) // 2:
                raise RuntimeError("simulated decoder bug")
            if mode == "back" and i == len(spec) // 2:
                yield Fraction(0), _RawImg(boxes, (mp4, i)), 0
                continue
            yield Fraction(t), _RawImg(boxes, (mp4, i)), 0

    run.iter_raw_frames, run.frame_to_bgr = gen, (lambda vf, k: vf)
    try:
        files = sorted(tmp.glob("*.parquet"))
        runner = run.run_threaded if kind == "threaded" else run.run_serial
        res = runner(files, None, det, w, True, "staged-1>3>all-VPN")
    finally:
        run.iter_raw_frames, run.frame_to_bgr = orig
    return w.lines(), det.seen, res


def test_threaded_equals_serial_exactly():
    import tempfile
    from pathlib import Path as P_
    saved = run.QUEUE_BYTES, run.QUEUE_CHUNK_FRAMES
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = P_(d)
            specs = _pipeline_inputs(tmp)
            s_lines, s_seen, s_res = _run_pipeline("serial", tmp, specs)
            for qb, chunk in ((512 << 20, 32), (1, 3), (1, 1)):  # normal, and a tiny queue (back-pressure)
                run.QUEUE_BYTES, run.QUEUE_CHUNK_FRAMES = qb, chunk
                t_lines, t_seen, t_res = _run_pipeline("threaded", tmp, specs)
                assert t_lines == s_lines, f"queue {qb}/{chunk}: output differs"
                assert t_seen == s_seen, f"queue {qb}/{chunk}: different frames or order sent to the model"
                assert t_res == s_res, (t_res, s_res)
            errs_ = [r for r in s_lines if "error" in r]
            assert len(errs_) == 5, errs_  # fail, back, exc, empty, missing
            assert len(s_res[2]) == 1  # c.parquet unreadable
    finally:
        run.QUEUE_BYTES, run.QUEUE_CHUNK_FRAMES = saved


def test_threaded_decoder_crash_does_not_hang():
    import tempfile
    import threading
    from pathlib import Path as P_
    with tempfile.TemporaryDirectory() as d:
        tmp = P_(d)
        specs = _pipeline_inputs(tmp)
        out = {}

        def target():
            try:
                _run_pipeline("threaded", tmp, specs, crash_key=b"v5")
                out["r"] = "returned"
            except RuntimeError as e:
                out["r"] = f"raised {e}"

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(timeout=30)
        assert not th.is_alive(), "threaded pipeline hung after a decoder-thread crash"
        assert out["r"].startswith("raised decoder thread failed"), out


def test_uniform_positions_match_simulator_float_rule():
    for n in range(0, 400):
        for k in (1, 3, 5, 9, 15):
            want = list(range(n)) if n <= k else sorted({int((j + 0.5) * n / k) for j in range(k)})
            assert run.uniform_positions(n, k) == want, (n, k)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"== {len(tests)} tests, {failed} failed")
    sys.exit(1 if failed else 0)
