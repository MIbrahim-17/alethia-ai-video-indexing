# Video Indexing at Speed

`run.py` indexes FineVideo Parquet files. For every second of every video, it writes which of person, car, dog, laptop and cell phone are present and how many people are visible. It uses YOLO11l (imgsz 640, conf 0.4, class-aware NMS at IoU 0.7), following the brief's rules.

## Install and run

No `apt-get` packages are needed: PyAV ships its own FFmpeg, and OpenCV is the headless build. Tested with Python 3.11 and an NVIDIA driver with CUDA 12 support.

```bash
git clone <this repository> && cd <repository>
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Preparation (untimed; needs internet; run once on the machine that does the timed run)

```bash
python prep.py
```

`prep.py` does three things:

1. Downloads the official `yolo11l.pt` into `weights/` and checks its sha256.
2. Builds a **TensorRT FP16 engine**, `weights/yolo11l.engine`, for this GPU: dynamic shapes, batch 1–32, imgsz 640.
3. Checks that the engine loads and runs a batch of 32.

A TensorRT engine only works on the GPU model and TensorRT version it was built with, so it has to be built on the grading machine. Building took about 6 minutes on our Quadro RTX 6000.

`requirements.txt` pulls about 9 GB of wheels. These include `tensorrt-cu12-libs` (4 GB) from NVIDIA's package index, listed at the top of the file, and the CUDA 12.6 builds of `torch` and `torchvision`, pinned as exact wheel URLs with sha256.

### Timed run (offline)

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

- Nothing needs the network: `run.py` sets `YOLO_OFFLINE=1` and turns off Ultralytics' auto-install.
- When the run finishes, a `DONE ...` summary line goes to stderr, with counts, the backend used and the wall time.
- If `weights/yolo11l.engine` is missing or fails to load, `run.py` prints a warning and carries on with PyTorch FP16. It's slower, but the results agree with the engine's to within the accuracy figures below.

### Output

There is one JSON line per second of each video, in input order:

```json
{"video": "train-00000-of-01357.parquet:0", "second": 3, "objects": ["person", "laptop"], "people": 2}
```

- **`video`** is `<parquet file name>:<row index>`.
- **`second`** is counted from the first decoded frame.
- **`objects`** is in the fixed order person, car, dog, laptop, cell phone.
- **Seconds with no frames** still get a line, with `"objects": []` and `"people": 0`: every second from 0 to the one holding the last frame is written. This happens when there is a timestamp gap inside a video.

If a file or video can't be read, decoding stops partway through, or the GPU runs out of memory on a single frame, the run carries on. For a video, it writes an `{"video": ..., "error": ...}` line and keeps the seconds it had finished. A Parquet file that can't be read at all is reported as a `WARNING` on stderr, and the run moves on to the next file.

## How it works

Each choice is described, with its measurements and the alternatives, in [`DECISIONS.md`](DECISIONS.md). The table below gives the entry for each part.

| Part | Choice | Entry |
|---|---|---|
| Decoding | PyAV, in a separate thread that overlaps with inference. A bounded queue of 512 MB of decoded frames sits between decoding and inference. | D-023 |
| Frame sampling | **Staged, `staged-1>3>all-VPN`**, described below the table. | D-016, D-017 |
| Model | YOLO11l as a TensorRT FP16 engine. Batches hold up to 32 frames of the same size, using Ultralytics' letterbox and NMS unchanged. | D-022, D-025 |
| Aggregation | An object is present if it is found in at least half of the frames run for that second. People is the floor of the median count over those frames. | D-015, D-017 |
| Memory | Decoded frames are bounded: chunks of 32 seconds or 1 GB, and a second larger than 512 MB streams straight through the model. On CUDA out-of-memory, the batch is halved and retried. | D-019 |

How the staged sampling decides which frames to run:
- **First, the middle frame** of each second goes through the model.
- **If that second's result differs from either neighbouring second's,** the model also runs on three evenly spaced frames.
- **If those frames still disagree** on which objects are present or on the people count, every frame in that second goes through the model.

Against an every-frame FP32 reference (dev set: 101 videos, 27,388 seconds; untouched holdout: 48 videos, 14,341 seconds), the shipped configuration scores:

| Set | F1 (macro gate) | People exact | People within ±1 |
|---|---|---|---|
| dev | 0.9928 | 0.9922 | 0.9998 |
| holdout | 0.9933 | 0.9934 | 0.9999 |

The brief's thresholds are F1 ≥ 0.95, exact ≥ 0.85 and within ±1 ≥ 0.95.

TensorRT doesn't build engines deterministically: each build may choose different kernels, which moves a few detections at the 0.4 confidence threshold. In a clean-environment test, a second engine built with `prep.py` changed 0.12% of seconds, and the dev F1 was 0.9918, with every metric still above the targets. Expect the same size of difference between `results.jsonl.gz` and a run on the grading machine (D-027).

For speed, the full 50 files (about 128 h of video) took **1 h 58 min on one Quadro RTX 6000** (see [`HARDWARE.md`](HARDWARE.md)). The speed log at the end of `DECISIONS.md` records each step.

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | The script, which is a single file. |
| `prep.py` | The preparation step. |
| `requirements.txt` | Exact versions of every package. |
| `results.jsonl.gz` | Results for all 50 files: `train-00000` to `train-00049`. |
| `HARDWARE.md`, `WRITEUP.md` | The hardware used, and the answers to the brief's write-up questions. |
| `DECISIONS.md` | Every design decision, with alternatives, measurements and a speed log. |
| `validation/` | The development and validation tools, listed in the table below. They aren't needed to run `run.py`. |
| `ai_chats/` | The full AI chat history. |
| `tools/` | Helper scripts for syncing to the development server. |

The tools in `validation/` are:

| Tool | What it does |
|---|---|
| `build_reference.py`, `reference_aggregate.py`, `check_reference.py` | Build and check the every-frame reference. |
| `score.py` | Scores results against a reference using the brief's metrics. |
| `simulate_sampling.py`, `compare_sim.py` | Predict which frames the staged sampling runs, and check a run against that prediction exactly. |
| `fixtures/` | 17 synthetic damaged or unusual videos and their checks. |
| `test_run_unit.py` | Unit tests. |
| `test_memory.py` | Memory-bound tests. |
| `offline_guard.py` | Runs `run.py` with Python networking blocked. |
| `check_full_results.py` | An independent completeness check of a full results file. |
