"""Score a results.jsonl file against the every-frame reference (brief §5, D-015 #4).

Metrics:
  - objects F1: micro (pooled), macro (mean of per-class F1), per class, and
    sample-averaged (for information only; an empty-vs-empty second scores 1.0).
    The pass gate uses min(micro, macro).
  - people: the share of seconds that match exactly, and the share within ±1.
A reference second missing from the output counts as wrong: no objects and a wrong
people count. Duplicate lines, or lines the reference doesn't have, fail the
completeness check outright.

Usage:
  python validation/score.py --ref work/reference/reference.jsonl --pred results.jsonl \
      [--json-out score.json] [--worst 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]
THRESH = {"f1": 0.95, "people_exact": 0.85, "people_within1": 0.95}


def load(path):
    secs, errs, dups, bad = {}, {}, [], []
    for n, line in enumerate(open(path), 1):
        r = json.loads(line)
        if "error" in r:
            errs[r["video"]] = r["error"]
            continue
        key = (r["video"], r["second"])
        if key in secs:
            dups.append(key)
        if (not isinstance(r.get("objects"), list) or any(o not in TARGETS for o in r["objects"])
                or not isinstance(r.get("people"), int)):
            bad.append(n)
        secs[key] = (frozenset(r["objects"]), r["people"])
    return secs, errs, dups, bad


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 1.0 if d == 0 else 2 * tp / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json-out")
    ap.add_argument("--worst", type=int, default=10)
    a = ap.parse_args()

    ref, ref_err, _, _ = load(a.ref)
    pred, pred_err, dups, bad = load(a.pred)
    extra = [k for k in pred if k not in ref]
    missing = [k for k in ref if k not in pred]

    tp, fp, fn = Counter(), Counter(), Counter()
    exact = within1 = 0
    sample_f1 = 0.0
    per_video = defaultdict(lambda: Counter())
    for key, (robj, rpeople) in ref.items():
        pobj, ppeople = pred.get(key, (frozenset(), None))
        for c in TARGETS:
            tp[c] += (c in robj) and (c in pobj)
            fp[c] += (c not in robj) and (c in pobj)
            fn[c] += (c in robj) and (c not in pobj)
        s_tp = len(robj & pobj)
        sample_f1 += f1(s_tp, len(pobj - robj), len(robj - pobj))
        ok_exact = ppeople == rpeople
        ok_w1 = ppeople is not None and abs(ppeople - rpeople) <= 1
        exact += ok_exact
        within1 += ok_w1
        v = per_video[key[0]]
        v["seconds"] += 1
        v["people_exact"] += ok_exact
        v["obj_err"] += len(robj ^ pobj)

    n = len(ref)
    per_class = {c: f1(tp[c], fp[c], fn[c]) for c in TARGETS}
    micro = f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro_classes = [c for c in TARGETS if tp[c] + fp[c] + fn[c] > 0]
    macro = sum(per_class[c] for c in macro_classes) / max(len(macro_classes), 1)
    res = {
        "seconds_ref": n,
        "seconds_pred": len(pred),
        "missing_seconds": len(missing),
        "extra_seconds": len(extra),
        "duplicate_lines": len(dups),
        "malformed_lines": len(bad),
        "error_videos_pred": len(pred_err),
        "error_videos_ref": len(ref_err),
        "f1_micro": micro,
        "f1_macro": macro,
        "f1_gate": min(micro, macro),
        "f1_sample_avg_info_only": sample_f1 / max(n, 1),
        "f1_per_class": per_class,
        "class_counts": {c: {"tp": tp[c], "fp": fp[c], "fn": fn[c]} for c in TARGETS},
        "people_exact": exact / max(n, 1),
        "people_within1": within1 / max(n, 1),
    }
    complete = not (extra or dups or bad)
    res["complete"] = complete
    res["pass"] = (complete and res["f1_gate"] >= THRESH["f1"]
                   and res["people_exact"] >= THRESH["people_exact"]
                   and res["people_within1"] >= THRESH["people_within1"])

    print(f"seconds ref={n} pred={len(pred)} missing={len(missing)} extra={len(extra)} "
          f"dups={len(dups)} malformed={len(bad)} errors(pred)={len(pred_err)}")
    print(f"F1 micro={micro:.4f} macro={macro:.4f} gate={res['f1_gate']:.4f} "
          f"(sample-avg {res['f1_sample_avg_info_only']:.4f}, info only)")
    print("F1 per class: " + "  ".join(f"{c}={per_class[c]:.4f}" for c in TARGETS))
    print(f"people exact={res['people_exact']:.4f} within1={res['people_within1']:.4f}")
    print(f"PASS={res['pass']}  (gate f1>={THRESH['f1']}, exact>={THRESH['people_exact']}, "
          f"within1>={THRESH['people_within1']}, complete={complete})")
    if a.worst:
        worst = sorted(per_video.items(), key=lambda kv: kv[1]["people_exact"] / kv[1]["seconds"])
        print(f"worst {a.worst} videos by people-exact:")
        for vid, v in worst[: a.worst]:
            print(f"  {vid}: exact {v['people_exact'] / v['seconds']:.3f} "
                  f"obj_err {v['obj_err']} over {v['seconds']} s")
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
