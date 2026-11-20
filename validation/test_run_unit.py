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
