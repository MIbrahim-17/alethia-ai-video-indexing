"""Simulate frame-sampling strategies on the every-frame reference (D-016).

The reference stores every frame's detections (D-014), and batched and single-frame
inference give the same detections there. So the output of any sampling strategy can
be computed exactly from those stored detections: pick the sampled frames, aggregate
them with the brief §5 rules, and score against the every-frame reference. No GPU is
needed. The cost of a strategy is the share of frames that go through the model.

Strategies (n = number of frames in the second, in time order, k = frames per second):
  uniform-k      k frames at evenly spaced positions in the second, by index:
                 positions floor((j + 0.5) * n / k) for j = 0..k-1. All frames if n <= k.
  time-k         the frame nearest each time s + (j + 0.5) / k (the same as uniform-k for
                 constant frame rate; included to check variable frame rate).
  stride-N       every Nth frame of the video (global index % N == 0). A second that gets
                 no frame uses its middle frame, so no second is empty.
  adapt-k-F      uniform-k first. If the second is flagged as uncertain by any of the
                 criteria in F, run the model on all its frames. Criteria:
                   V  the sampled frames disagree on whether some object is present
                   P  the sampled frames disagree on the person count
                   C  a sampled frame has a target detection with confidence < C_LO
                   N  the second's sampled result (objects, people) differs from the
                      previous or next second's sampled result

Metrics are pooled over all seconds of all 101 videos (as the brief's grader is assumed
to do) and also given as the 5th percentile of a bootstrap over videos (layer 5), and for
the sequential and diverse parts separately.

Usage (on the server):
  python validation/simulate_sampling.py --ref work/reference --out work/sampling/sim.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]
C_LO = 0.5
THRESH = {"f1": 0.95, "people_exact": 0.85, "people_within1": 0.95}


# ---------------------------------------------------------------- data

def load_reference(ref_dir: Path):
    videos = []
    for meta in sorted((ref_dir / "frames").glob("*.json")):
        info = json.loads(meta.read_text())
        t = pq.read_table(meta.with_suffix(".parquet"), columns=["second", "t_rel", "det_cls", "det_conf"])
        sec = np.array(t.column("second").to_pylist(), dtype=np.int64)
        trel = np.array(t.column("t_rel").to_pylist(), dtype=np.float64)
        cls = t.column("det_cls").to_pylist()
        conf = t.column("det_conf").to_pylist()
        pres = np.zeros((len(sec), 5), dtype=bool)
        cnt = np.zeros(len(sec), dtype=np.int64)
        minconf = np.ones(len(sec))
        for i, (c, f) in enumerate(zip(cls, conf)):
            for k in c:
                pres[i, k] = True
            cnt[i] = c.count(0)
            if f:
                minconf[i] = min(f)
        n_sec = int(sec.max()) + 1
        bounds = np.searchsorted(sec, np.arange(n_sec + 1))  # frames of second s: bounds[s]:bounds[s+1]
        videos.append({"video": info["video"], "sec": sec, "t": trel, "pres": pres, "cnt": cnt,
                       "minconf": minconf, "bounds": bounds, "n_sec": n_sec,
                       "part": "seq" if info["video"].startswith(("train-00000-", "train-00001-")) else "div"})
    return videos


def agg(v, idx):
    """brief §5 on the frames idx: (presence bool[5], people)."""
    if len(idx) == 0:
        return np.zeros(5, bool), 0
    p = 2 * v["pres"][idx].sum(0) >= len(idx)
    return p, math.floor(float(np.median(v["cnt"][idx])))


# ---------------------------------------------------------------- strategies
# Each returns, per video, a list of index arrays (one per second) and the frame count used.

def uniform_idx(lo, hi, k):
    n = hi - lo
    if n <= k:
        return np.arange(lo, hi)
    return lo + np.unique(((np.arange(k) + 0.5) * n / k).astype(np.int64))


def strat_uniform(v, k):
    b = v["bounds"]
    return [uniform_idx(b[s], b[s + 1], k) for s in range(v["n_sec"])]


def strat_time(v, k):
    b, t = v["bounds"], v["t"]
    out = []
    for s in range(v["n_sec"]):
        lo, hi = b[s], b[s + 1]
        if hi - lo <= k:
            out.append(np.arange(lo, hi))
            continue
        tt = t[lo:hi]
        targets = s + (np.arange(k) + 0.5) / k
        out.append(lo + np.unique(np.abs(tt[None, :] - targets[:, None]).argmin(1)))
    return out


def strat_stride(v, n):
    b = v["bounds"]
    out = []
    for s in range(v["n_sec"]):
        lo, hi = b[s], b[s + 1]
        first = lo + (-lo) % n
        idx = np.arange(first, hi, n)
        if len(idx) == 0 and hi > lo:
            idx = np.array([(lo + hi - 1) // 2])
        out.append(idx)
    return out


def strat_adapt(v, k, flags):
    b = v["bounds"]
    base = strat_uniform(v, k)
    res = [agg(v, idx) for idx in base]
    out = []
    for s, idx in enumerate(base):
        lo, hi = b[s], b[s + 1]
        unsure = False
        if len(idx) < hi - lo:
            pr = v["pres"][idx]
            if "V" in flags and (pr.any(0) != pr.all(0)).any():
                unsure = True
            if "P" in flags and v["cnt"][idx].min() != v["cnt"][idx].max():
                unsure = True
            if "C" in flags and v["minconf"][idx].min() < C_LO:
                unsure = True
            if "N" in flags:
                for o in (s - 1, s + 1):
                    if 0 <= o < len(res) and ((res[o][0] != res[s][0]).any() or res[o][1] != res[s][1]):
                        unsure = True
        out.append(np.arange(lo, hi) if unsure else idx)
    return out


def unsure_flags(v, idx, flags, c_lo):
    pr = v["pres"][idx]
    if "V" in flags and (pr.any(0) != pr.all(0)).any():
        return True
    if "P" in flags and v["cnt"][idx].min() != v["cnt"][idx].max():
        return True
    if "C" in flags and v["minconf"][idx].min() < c_lo:
        return True
    return False


def strat_staged(v, stages, flags, c_lo=C_LO):
    """Staged escalation: uniform-k1; if unsure, add uniform-k2 frames; ...; last stage may be 0 (= all).

    The aggregation uses every frame inferred so far (the union). With k2 = 3 * k1 the uniform
    positions nest, so no inferred frame is wasted. N (neighbours) is checked on stage 1 only.
    """
    b = v["bounds"]
    first = strat_uniform(v, stages[0])
    res = [agg(v, idx) for idx in first]
    out = []
    for s, idx in enumerate(first):
        lo, hi = b[s], b[s + 1]
        unsure = unsure_flags(v, idx, flags, c_lo) if len(idx) < hi - lo else False
        if "N" in flags and len(idx) < hi - lo:
            for o in (s - 1, s + 1):
                if 0 <= o < len(res) and ((res[o][0] != res[s][0]).any() or res[o][1] != res[s][1]):
                    unsure = True
        for k in stages[1:]:
            if not unsure or len(idx) == hi - lo:
                break
            nxt = np.arange(lo, hi) if k == 0 else uniform_idx(lo, hi, k)
            idx = np.union1d(idx, nxt)
            unsure = unsure_flags(v, idx, flags.replace("N", ""), c_lo) if len(idx) < hi - lo else False
        out.append(idx)
    return out


# ---------------------------------------------------------------- scoring

def video_counts(v, sel):
    """Per video: tp/fp/fn per class (5x3), exact, within1, seconds, frames used, seconds escalated."""
    ref_p = np.zeros((v["n_sec"], 5), bool)
    ref_c = np.zeros(v["n_sec"], np.int64)
    got_p = np.zeros_like(ref_p)
    got_c = np.zeros_like(ref_c)
    b = v["bounds"]
    for s in range(v["n_sec"]):
        ref_p[s], ref_c[s] = agg(v, np.arange(b[s], b[s + 1]))
        got_p[s], got_c[s] = agg(v, sel[s])
    tpfpfn = np.stack([(ref_p & got_p).sum(0), (~ref_p & got_p).sum(0), (ref_p & ~got_p).sum(0)], 1)
    other = np.array([(ref_c == got_c).sum(), (np.abs(ref_c - got_c) <= 1).sum(), v["n_sec"],
                      sum(len(x) for x in sel), len(v["sec"])])
    return tpfpfn, other


def metrics(tpfpfn, other):
    tp, fp, fn = tpfpfn[..., 0], tpfpfn[..., 1], tpfpfn[..., 2]
    d = 2 * tp + fp + fn
    per = np.where(d == 0, 1.0, 2 * tp / np.maximum(d, 1))
    T, FP, FN = tp.sum(-1), fp.sum(-1), fn.sum(-1)
    micro = 2 * T / np.maximum(2 * T + FP + FN, 1)
    return {"micro_f1": micro, "macro_f1": per.mean(-1), "f1_gate": np.minimum(micro, per.mean(-1)),
            "people_exact": other[..., 0] / other[..., 2], "people_within1": other[..., 1] / other[..., 2],
            "per_class": per, "frac_frames": other[..., 3] / other[..., 4]}


def evaluate(videos, sel_fn, rng, n_boot=2000):
    counts = [video_counts(v, sel_fn(v)) for v in videos]
    A = np.stack([c[0] for c in counts]).astype(np.float64)  # V x 5 x 3
    O = np.stack([c[1] for c in counts]).astype(np.float64)  # V x 5
    pooled = metrics(A.sum(0), O.sum(0))
    res = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in pooled.items()}
    res["per_class"] = dict(zip(TARGETS, np.round(pooled["per_class"], 4).tolist()))
    W = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)), size=n_boot).astype(np.float64)
    boot = metrics(np.einsum("bv,vcj->bcj", W, A), W @ O)
    for k in ("f1_gate", "people_exact", "people_within1"):
        res[f"{k}_p05"] = float(np.percentile(boot[k], 5))
    for part in ("seq", "div"):
        m = np.array([v["part"] == part for v in videos])
        pm = metrics(A[m].sum(0), O[m].sum(0))
        res[part] = {k: round(float(pm[k]), 4) for k in ("f1_gate", "people_exact", "people_within1")}
    res["worst_video_exact"] = float((O[:, 0] / O[:, 2]).min())
    res["passes"] = bool(res["f1_gate"] >= THRESH["f1"] and res["people_exact"] >= THRESH["people_exact"]
                         and res["people_within1"] >= THRESH["people_within1"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="work/reference")
    ap.add_argument("--out", default="work/sampling/sim.json")
    ap.add_argument("--only-staged", action="store_true")
    a = ap.parse_args()
    videos = load_reference(Path(a.ref))
    print(f"loaded {len(videos)} videos, {sum(len(v['sec']) for v in videos)} frames, "
          f"{sum(v['n_sec'] for v in videos)} seconds")
    # Self-check: this file's every-frame aggregation must reproduce reference.jsonl exactly.
    ref_lines = {}
    for line in open(Path(a.ref) / "reference.jsonl"):
        r = json.loads(line)
        if "second" in r:
            ref_lines[(r["video"], r["second"])] = (r["objects"], r["people"])
    mine = {}
    for v in videos:
        b = v["bounds"]
        for s in range(v["n_sec"]):
            p, c = agg(v, np.arange(b[s], b[s + 1]))
            mine[(v["video"], s)] = ([TARGETS[k] for k in range(5) if p[k]], c)
    assert mine == ref_lines, "simulator aggregation differs from reference.jsonl"
    print(f"self-check: {len(mine)} seconds identical to reference.jsonl")

    strategies = {"all": lambda v: [np.arange(v["bounds"][s], v["bounds"][s + 1]) for s in range(v["n_sec"])]}
    for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15):
        strategies[f"uniform-{k}"] = lambda v, k=k: strat_uniform(v, k)
    for k in (1, 2, 3, 5):
        strategies[f"time-{k}"] = lambda v, k=k: strat_time(v, k)
    for n in (2, 3, 5, 8, 10, 15, 30):
        strategies[f"stride-{n}"] = lambda v, n=n: strat_stride(v, n)
    for k in (1, 2, 3, 4, 5):
        for flags in ("V", "VP", "VPC", "N", "VPN", "VPCN"):
            if k == 1 and flags in ("V", "VP"):
                continue  # one frame can't disagree with itself
            strategies[f"adapt-{k}-{flags}"] = lambda v, k=k, f=flags: strat_adapt(v, k, f)

    for stages in ((1, 3, 0), (1, 3, 9, 0), (3, 9, 0), (3, 9), (3, 9, 27, 0), (5, 15, 0), (1, 3, 9)):
        for flags in ("VP", "VPC", "VPN", "VPCN"):
            name = "staged-" + ">".join("all" if k == 0 else str(k) for k in stages) + "-" + flags
            strategies[name] = lambda v, st=stages, f=flags: strat_staged(v, st, f)
    for c_lo in (0.45, 0.6, 0.7):
        for stages in ((1, 3, 9, 0), (3, 9, 0)):
            name = "staged-" + ">".join("all" if k == 0 else str(k) for k in stages) + f"-VPC{c_lo}"
            strategies[name] = lambda v, st=stages, c=c_lo: strat_staged(v, st, "VPC", c)
    for stages in ((1, 3, 9), (1, 3, 9, 0), (3, 9), (3, 9, 0), (1, 5, 15), (5, 15)):
        for flags in ("V", "VC", "VN", "VCN"):
            name = "staged-" + ">".join("all" if k == 0 else str(k) for k in stages) + "-" + flags
            strategies[name] = lambda v, st=stages, f=flags: strat_staged(v, st, f)
    if a.only_staged:
        strategies = {k: f for k, f in strategies.items() if k.startswith("staged") or k == "all"}

    results = {}
    hdr = f"{'strategy':16s} {'frames':>7s} {'F1gate':>7s} {'p05':>7s} {'exact':>7s} {'p05':>7s} {'within1':>7s} {'p05':>7s} {'seq ex':>7s} {'div ex':>7s} pass"
    print(hdr)
    for name, fn in strategies.items():
        r = evaluate(videos, fn, np.random.default_rng(20261120))
        results[name] = r
        print(f"{name:16s} {r['frac_frames']:7.3f} {r['f1_gate']:7.4f} {r['f1_gate_p05']:7.4f} "
              f"{r['people_exact']:7.4f} {r['people_exact_p05']:7.4f} {r['people_within1']:7.4f} "
              f"{r['people_within1_p05']:7.4f} {r['seq']['people_exact']:7.4f} {r['div']['people_exact']:7.4f} "
              f"{'yes' if r['passes'] else 'NO'}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
