"""Build the every-frame reference for the frozen validation subset (D-003, D-014).

It runs YOLO11l on every decoded frame, with the brief's settings passed explicitly,
and stores each frame's detections for the five target classes in one Parquet file per
video. The per-second reference is computed from these files by
`validation/reference_aggregate.py`.

Correctness comes before speed here. The code is kept deliberately simple and
independent of `run.py`, so a bug in one is unlikely to repeat in the other.

Usage (on the server):
  python validation/build_reference.py --subset validation/subset/frozen_subset.json \
      --out work/reference --gpus 0,1
Rerunning resumes: videos that already have output are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from fractions import Fraction
from pathlib import Path

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]  # brief §1, exact names
PREDICT_KW = dict(imgsz=640, conf=0.4, iou=0.7, verbose=False)  # brief §3; iou must be explicit (D-014)


def safe_name(video_id: str) -> str:
    return video_id.replace(":", "__")


# ---------------------------------------------------------------- decoding

def decode_frames(mp4: bytes, out_q: queue.Queue, info: dict):
    """Decode every frame in presentation order and put (pts, bgr_array) items on out_q.

    The queue ends with None. Rotation from the display matrix is applied to each frame.
    Any decoding failure is stored in info["error"], and frames already sent are kept.
    """
    import io

    import av
    import numpy as np

    try:
        with av.open(io.BytesIO(mp4), mode="r") as c:
            if not c.streams.video:
                raise ValueError("no video stream")
            s = c.streams.video[0]
            info["time_base"] = [s.time_base.numerator, s.time_base.denominator]
            info["codec"] = s.codec_context.name
            for f in c.decode(s):
                if f.pts is None:
                    raise ValueError("frame without pts")
                rot = int(f.rotation or 0)  # degrees counterclockwise, in [-180, 180]
                img = f.to_ndarray(format="bgr24")
                if rot % 360:
                    # The display matrix rotates `rot` degrees counterclockwise, and
                    # np.rot90 with k=1 rotates 90 degrees counterclockwise. The direction
                    # is checked by the D-013 rotation fixtures.
                    img = np.ascontiguousarray(np.rot90(img, k=(rot // 90) % 4))
                info.setdefault("rotation", rot)
                info.setdefault("shape_hw", list(img.shape[:2]))
                out_q.put((f.pts, img))
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        info["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        out_q.put(None)


# ---------------------------------------------------------------- GPU worker

def gpu_worker(gpu: str, task_q, done_q, out_dir: str, weights: str, batch: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # set before torch is imported
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    name_to_id = {v: k for k, v in model.names.items()}
    target_ids = {name_to_id[n]: i for i, n in enumerate(TARGETS)}  # coco id -> 0..4
    gpu_name = torch.cuda.get_device_name(0)
    frames_dir = Path(out_dir) / "frames"

    while True:
        task = task_q.get()
        if task is None:
            break
        vid, mp4 = task
        t0 = time.time()
        info: dict = {"video": vid, "gpu": f"{gpu}:{gpu_name}"}
        rows = {"frame_idx": [], "pts": [], "t_rel": [], "second": [],
                "det_cls": [], "det_conf": [], "det_xyxy": []}
        try:
            fq: queue.Queue = queue.Queue(maxsize=256)
            th = threading.Thread(target=decode_frames, args=(mp4, fq, info), daemon=True)
            th.start()
            first_pts = None
            tb = None
            idx = 0
            buf_pts, buf_img = [], []

            def flush():
                nonlocal idx
                if not buf_img:
                    return
                results = model.predict(buf_img, device=0, **PREDICT_KW)
                for p, r in zip(buf_pts, results):
                    b = r.boxes
                    cls = b.cls.cpu().numpy().astype(int)
                    conf = b.conf.cpu().numpy()
                    xyxy = b.xyxy.cpu().numpy()
                    keep = [i for i, c in enumerate(cls) if c in target_ids]
                    t_rel = (p - first_pts) * tb  # exact Fraction
                    rows["frame_idx"].append(idx)
                    rows["pts"].append(int(p))
                    rows["t_rel"].append(float(t_rel))
                    rows["second"].append(int(t_rel // 1))  # floor, exact
                    rows["det_cls"].append([target_ids[int(cls[i])] for i in keep])
                    rows["det_conf"].append([float(conf[i]) for i in keep])
                    rows["det_xyxy"].append([float(v) for i in keep for v in xyxy[i]])
                    idx += 1
                buf_pts.clear()
                buf_img.clear()

            while True:
                item = fq.get()
                if item is None:
                    break
                p, img = item
                if first_pts is None:
                    first_pts = p
                    tb = Fraction(*info["time_base"])
                buf_pts.append(p)
                buf_img.append(img)
                if len(buf_img) >= batch:
                    flush()
            flush()
            th.join()
        except Exception as e:  # noqa: BLE001
            info["error"] = info.get("error") or f"worker {type(e).__name__}: {e}"[:300]
            info["traceback"] = traceback.format_exc()[-2000:]

        info["n_frames"] = len(rows["frame_idx"])
        info["first_pts"] = first_pts if rows["pts"] else None
        info["elapsed_s"] = round(time.time() - t0, 3)
        table = pa.table({
            "frame_idx": pa.array(rows["frame_idx"], pa.int32()),
            "pts": pa.array(rows["pts"], pa.int64()),
            "t_rel": pa.array(rows["t_rel"], pa.float64()),
            "second": pa.array(rows["second"], pa.int32()),
            "det_cls": pa.array(rows["det_cls"], pa.list_(pa.int8())),
            "det_conf": pa.array(rows["det_conf"], pa.list_(pa.float32())),
            "det_xyxy": pa.array(rows["det_xyxy"], pa.list_(pa.float32())),
        })
        stem = frames_dir / safe_name(vid)
        pq.write_table(table, f"{stem}.parquet.tmp")
        Path(f"{stem}.json.tmp").write_text(json.dumps(info))
        os.replace(f"{stem}.parquet.tmp", f"{stem}.parquet")
        os.replace(f"{stem}.json.tmp", f"{stem}.json")  # the .json file marks the video as done
        done_q.put((vid, info["n_frames"], info["elapsed_s"], info.get("error")))


# ---------------------------------------------------------------- main

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--weights", default="weights/yolo11l.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (smoke test)")
    a = ap.parse_args()

    subset = json.loads(Path(a.subset).read_text())
    for name, meta in subset["files"].items():  # refuse to run on data that changed
        got = sha256(Path(meta["dir"]) / name)
        if got != meta["sha256"]:
            raise SystemExit(f"sha256 mismatch for {name}: {got}")
    videos = subset["videos"][: a.limit or None]
    frames_dir = Path(a.out) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    todo = [v for v in videos if not (frames_dir / f"{safe_name(v['video'])}.json").exists()]
    print(f"{len(videos)} videos in subset, {len(todo)} to do", flush=True)

    import pyarrow.parquet as pq

    ctx = mp.get_context("spawn")
    task_q, done_q = ctx.Queue(maxsize=4), ctx.Queue()
    gpus = a.gpus.split(",")
    procs = [ctx.Process(target=gpu_worker, args=(g, task_q, done_q, a.out, a.weights, a.batch))
             for g in gpus]
    for p in procs:
        p.start()

    def feeder():
        by_file: dict = {}
        for v in todo:
            by_file.setdefault(v["file"], set()).add(v["row"])
        for fname, rows in sorted(by_file.items()):
            meta = subset["files"][fname]
            pf = pq.ParquetFile(Path(meta["dir"]) / fname)
            r = 0
            for b in pf.iter_batches(batch_size=1, columns=["mp4"]):
                if r in rows:
                    task_q.put((f"{fname}:{r}", b.column("mp4")[0].as_py()))
                r += 1
        for _ in procs:
            task_q.put(None)

    threading.Thread(target=feeder, daemon=True).start()
    t0, n_done, n_frames = time.time(), 0, 0
    while n_done < len(todo):
        try:
            vid, nf, el, err = done_q.get(timeout=30)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                raise SystemExit("all GPU workers exited early")
            continue
        n_done += 1
        n_frames += nf
        dt = time.time() - t0
        print(f"[{n_done}/{len(todo)}] {vid} frames={nf} {el:.1f}s"
              f"{' ERROR ' + err if err else ''} | total {n_frames} frames "
              f"{n_frames / dt:.0f} fps {dt / 60:.1f} min", flush=True)
    for p in procs:
        p.join()
    print("done", flush=True)


if __name__ == "__main__":
    main()
