# Decisions

A record of the important decisions made for the Video Indexing take-home (see `AI_Engineering_TakeHome_Brief.pdf`).

**How to use this file**
- Read it before any significant architectural or technical change.
- Only add a decision after it has been discussed and agreed.
- Never delete or rewrite an existing decision. To change one, add a new decision and mark the old one `Superseded by D-XXX`.
- Statuses are `Accepted`, `Superseded` and `Rejected`.

---

## Fixed constraints (set by the brief, not open to decision)

These are requirements we can't change. They are listed so that any proposed decision can be checked against them.

| # | Constraint | Brief § |
|---|---|---|
| C-1 | Entry point is `python run.py --input <folder> --output results.jsonl` | 1 |
| C-2 | Model is Ultralytics YOLO11l, official `yolo11l.pt` weights, input size 640 | 3 |
| C-3 | A detection counts only if its confidence is 0.4 or higher. Everything else stays at the Ultralytics defaults: letterbox resize, class-aware NMS at IoU 0.7, BGR frames. | 3 |
| C-4 | Detect only `person`, `car`, `dog`, `laptop`, `cell phone`, using these exact names | 1, 4 |
| C-5 | Rotate frames that have a rotation tag before inference | 3 |
| C-6 | One JSONL line per second, covering second 0 through the second that holds the last frame. Timestamps are measured from the first decoded frame. ID format is `<parquet filename>:<row index>`. | 4 |
| C-7 | A damaged video gets one error line and processing continues. Lines already written for a video that fails partway are kept. | 1, 4 |
| C-8 | Passing requires objects F1 ≥ 0.95, people count exact on ≥ 85% of seconds, and people count within ±1 on ≥ 95% of seconds | 5 |
| C-9 | Grading machine is an RTX 4090 (24 GB), 32 threads, 64 GB RAM, Ubuntu 22.04, Python 3.11, CUDA 12. No Docker. The timed run is offline and stopped at 6 hours. The prep command is untimed and online. | 6 |
| C-10 | Deliverables are `run.py`, a pinned `requirements.txt`, `README.md`, `results.jsonl.gz`, `HARDWARE.md`, `WRITEUP.md` (written by hand, no AI) and an unedited `ai_chats/` | 7, 8 |

---

## Decision log

| ID | Title | Status | Date |
|---|---|---|---|
| D-001 | Development and benchmark server | Accepted | 2026-11-20 |
| D-002 | Choose options by timed runs, not by assumption | Accepted | 2026-11-20 |
| D-003 | Build the all-frames reference and baseline before any optimisation | Accepted | 2026-11-20 |
| D-004 | Five-layer validation framework | Accepted (layer details pending) | 2026-11-20 |
| D-005 | Git repository and secret handling | Accepted | 2026-11-20 |
| D-006 | Validation subset: about 100 videos | Accepted (updated 2026-11-20: diverse sample from outside the first 50 files) | 2026-11-20 |
| D-007 | GPU use: both GPUs for the reference, GPU 0 only for benchmarks | Accepted | 2026-11-20 |
| D-008 | Disk budget: 54 GB on the `ibrahim` root only | Accepted | 2026-11-20 |
| D-009 | Sudo policy | Accepted | 2026-11-20 |
| D-010 | Local repo is the source of truth, pushed to the server with rsync | Accepted | 2026-11-20 |
| D-011 | Server environment matched to the grading machine | Accepted | 2026-11-20 |
| D-012 | How the diverse sample was picked, and the resulting subset | Accepted (frozen 2026-11-20) | 2026-11-20 |
| D-013 | Synthetic fixture suite for unusual and damaged videos | Accepted | 2026-11-20 |
| D-014 | Every-frame reference builder design | Accepted (3 items pending confirmation) | 2026-11-20 |

### D-001: Development and benchmark server
- **Status:** Accepted · **Date:** 2026-11-20
- **Context:** We need an NVIDIA GPU for development, benchmarking and the full 50-file run (brief §9).
- **Decision:** Use the user's own server (`ibrahim@` over Tailscale) for all development and for the full run. The password is in `.env` as `VM_Password` and is never printed or committed.
- **Measured on 2026-11-20:**

  | Resource | Value |
  |---|---|
  | GPU | 2× Quadro RTX 6000, 24 GB each, Turing (compute capability 7.5). GPU 1 has about 10.7 GB in use by ollama. GPU 0 is idle. |
  | Driver | 595.99, CUDA 13.2 |
  | CPU | i9-14900K, 32 threads (the same thread count as the grading machine) |
  | RAM | 61 GB |
  | Disk | 54 GB free on the NVMe root. The second SSD (488 GB, NTFS) is mounted under another user. |
  | OS and software | Ubuntu 26.04, Python 3.14 only (no 3.11), `uv` available, no ffmpeg, sudo needs a password, internet access to Hugging Face works |
- **Alternatives considered:** Colab or Kaggle (time limits and small disks) and rented GPUs (cost). The user's own hardware avoids both.
- **Consequences:**
  - The GPU is a Turing card, not the grading machine's Ada RTX 4090. Absolute timings and the balance between decoding and inference won't carry over exactly, so they are only good for comparing options against each other.
  - TensorRT engines are tied to the GPU they were built on, so any engine has to be built in the prep step on the grading machine.
  - Disk space is tight: 54 GB free against about 25 GB of data plus the environment.
  - We need Python 3.11 (via `uv`) to match the grading machine.

### D-002: Choose options by timed runs, not by assumption
- **Status:** Accepted · **Date:** 2026-11-20
- **Context:** Several choices affect the timed run, and their real cost depends on the hardware and the data.
- **Decision:** Every choice that can affect the timed run gets benchmarked on real data before it's chosen. That covers decoding, frame sampling, the inference engine, concurrency and anything else of that kind. Each benchmark must also pass the validation suite (D-004). The result is recorded here as its own decision, with the measured numbers.
- **Alternatives considered:** Picking options from prior knowledge or published benchmarks. That's faster, but it's unreliable on this hardware and data, and it leaves nothing to support write-up questions 1, 3 and 4.
- **Consequences:** Every option needs a fair benchmark harness, meaning the same videos, warm-up runs and wall-clock timing. Development takes longer in exchange.

### D-003: Build the all-frames reference and baseline before any optimisation
- **Status:** Accepted · **Date:** 2026-11-20
- **Context:** Correctness is graded against an every-frame reference (brief §5), and speed counts only if the results pass.
- **Decision:**
  1. Build the test suite first, using a reference that runs the model on every frame with the brief's exact rules.
  2. Make the first baseline `run.py` use every frame too.
  3. Once the baseline's accuracy and timing are measured, move on to sampling, then the inference engine, decoding and the rest, one change at a time.
- **Alternatives considered:** Starting with an optimised, sampled pipeline. That would leave us with no trustworthy reference and no first-version timing for write-up question 1.
- **Consequences:** The baseline is slow. Its output is the ground truth that every later version gets scored against.

### D-004: Five-layer validation framework
- **Status:** Accepted at the level of the five layers. The checks inside each layer are still under discussion. · **Date:** 2026-11-20
- **Decision:** Validation has five layers:
  1. **Data and decoding correctness:** did we read every video and decode the right frames?
  2. **Model inference equivalence:** does YOLO receive equivalent pixels and produce equivalent detections?
  3. **Aggregation correctness:** are the votes, timestamps and medians calculated exactly?
  4. **JSONL completeness and scoring:** are all expected records present, and does the output meet the grading metrics?
  5. **Robustness and statistical validation:** will correctness hold up on unusual videos, after optimisation changes and on unseen data?
- **Alternatives considered:** Checking only the final score against the brief's thresholds. That shows *that* something is wrong but not *where*.
- **Consequences:** Every optimisation must pass all five layers before it's adopted (see D-002).

### D-005: Git repository and secret handling
- **Status:** Accepted · **Date:** 2026-11-20
- **Decision:**
  - The project is a git repo on branch `main`.
  - `.env` and `.env.*` are in `.gitignore`.
  - Secrets are read straight from `.env` by helper scripts and never echoed. That keeps them out of the AI chat logs, which have to be submitted unedited (brief §7).
- **Consequences:** Before every push, check that nothing is tracked under `.env` and search the chat logs for the password.

### D-006: Validation subset: about 100 videos
- **Status:** Accepted · **Date:** 2026-11-20
- **Context:** Running every frame of all 50 files would take an estimated 15 to 40 hours on this hardware, which doesn't fit the deadline (D-003).
- **Decision:** The reference and the baseline timing use about 100 videos in two parts:
  - **60 to 70 videos from two complete Parquet files.** These give realistic sequential I/O and processing performance.
  - **30 to 40 videos sampled from 8 to 12 other Parquet files.** These are chosen to cover a range of resolutions, frame rates, durations, codecs and orientations.
- **Alternatives considered:** Running every frame of all 50 files, which takes too long. A random subset only, which has poor coverage of edge cases. Complete files only, which have poor diversity.
- **Consequences:**
  - Timings for the full 50 files are extrapolated from the sequential part.
  - The full run on 50 files happens once, with the final version.
  - **Pending:** whether the diverse sample comes from inside the first 50 files or from outside them, which would better imitate the unseen test set.
- **Update 2026-11-20 (the pending point is now resolved):** The 30 to 40 diverse videos come from **outside the first 50 files** (index 50 and above). This imitates the unseen test set and avoids scoring against videos whose results we also submit. Rejected alternative: sampling from inside the first 50, which needs no extra disk but is a weaker test on unseen data. Extra disk cost is about 4 to 6 GB out of the D-008 budget.

### D-007: GPU use: both GPUs for the reference, GPU 0 only for benchmarks
- **Status:** Accepted · **Date:** 2026-11-20
- **Decision:**
  - Building the reference may use both RTX 6000s. Its output doesn't depend on speed.
  - All benchmarks and timed runs use GPU 0 only, set with `CUDA_VISIBLE_DEVICES=0`, to match the single GPU on the grading machine.
- **Consequences:** GPU 1 is shared with ollama (about 10.7 GB used), so reference jobs on GPU 1 must fit in the memory that's left.

### D-008: Disk budget: 54 GB on the `ibrahim` root only
- **Status:** Accepted · **Date:** 2026-11-20
- **Decision:** All work stays within the free space on `/`, about 54 GB, under `/home/ibrahim`. The `ai-security` NTFS volume must not be used.
- **Consequences:** The dataset (about 25 GB) and the environment (an estimated 6 to 8 GB) take most of the space. Intermediate files, such as decoded frames and per-frame detections, have to stay small or be cleaned up. Check free space before each large download.

### D-009: Sudo policy
- **Status:** Accepted · **Date:** 2026-11-20
- **Decision:** Sudo commands may use `VM_Password`, which is fed from `.env` and never echoed. **The user is told the exact command before every sudo call.** Pip-only dependencies are preferred, because every apt package also has to be listed in the README for the graders.

### D-010: Local repo is the source of truth, pushed to the server with rsync
- **Status:** Accepted · **Date:** 2026-11-20
- **Decision:**
  - Code is edited in the local git repo and pushed to `~/alethia` on the server with `tools/sync.sh`, which uses `rsync --delete`.
  - The sync leaves out `.env*`, `.git/`, `.venv/` and the data and work folders.
  - Remote commands go through `tools/ssh.sh`, and both scripts authenticate with `tools/askpass.sh`.
- **Alternatives considered:**
  - Editing on the server only: the local repo and the chat logs wouldn't line up.
  - A git remote on the server: an extra push and pull for every change.
- **Consequences:**
  - Results generated on the server, such as reference outputs and benchmark logs, have to be pulled back on purpose.
  - Secrets never reach the server through the sync. The server has its own `hf auth login`.
  - `weights/`, `*.pt` and `*.engine` are also left out of the sync and git. This keeps server-side model files safe from `--delete`. (Added 2026-11-20.)

### D-011: Server environment matched to the grading machine
- **Status:** Accepted · **Date:** 2026-11-20
- **Context:** The user asked for the server to match the grading environment as closely as possible (brief §6), so that optimisation results carry over.
- **Decision:**

  | Item | Choice |
  |---|---|
  | Python | 3.11.16, installed with `uv`, in `~/alethia/.venv` |
  | PyTorch | `torch==2.14.1+cu126`, `torchvision==0.29.1+cu126`, from the PyTorch cu126 index |
  | Other packages | `ultralytics==8.4.215`, `pyarrow==26.0.1`, `av==18.1.0`, `numpy==2.4.6`, `opencv-python==5.1.0.95` (pulled in by Ultralytics), cuDNN 9.10.2 |
  | Weights | Official `yolo11l.pt` from Ultralytics assets v8.4.0, sha256 `9ebd0e09…0023f`, stored in `weights/` |
- **Why the cu126 build:**
  - The grading driver only promises "CUDA 12". CUDA 12.6 needs an older driver than 12.8 or 12.9, so it has the widest compatibility.
  - cu126 also had the newest PyTorch release of the three CUDA 12 builds.
  - The build includes `sm_75` code for the RTX 6000 here. It includes `sm_86` code but not `sm_89`. NVIDIA GPUs can run code built for an earlier minor architecture of the same generation, so the `sm_86` code will run on the 4090 (`sm_89`).
- **Alternatives considered:**
  - `cu128` (only up to torch 2.11) and `cu129` (torch 2.13): both need newer drivers.
  - `cu13x` builds: excluded by the brief's CUDA 12.
  - Docker with an Ubuntu 22.04 image: the graders have no Docker, so it wouldn't be the same setup anyway.
- **Smoke test:** YOLO11l at size 640 with confidence 0.4 runs on GPU 0, at about 10.6 ms per frame (unbatched, FP32, random 720p input).
- **Known differences we accept:**
  - The OS is Ubuntu 26.04 rather than 22.04.
  - The GPU is a Turing RTX 6000 rather than an Ada 4090.
  - RAM is 61 GB rather than 64 GB.
- **Update 2026-11-20 (headless OpenCV):**
  - Replaced `ultralytics` and `opencv-python` with the official `ultralytics-opencv-headless==8.4.215`, which pulls in `opencv-python-headless==5.1.0.95`.
  - Reason: `opencv-python` needs the system library `libGL`, which a fresh Ubuntu 22.04 VM may not have, and then importing it fails. Checked with `ldd`: the headless `cv2` doesn't depend on libGL, X11 or Qt.
  - Rejected alternative: keeping `opencv-python` and listing `apt-get install libgl1` in the README. That adds a system dependency and a way for the run to fail.
  - `requirements.txt` must pin `ultralytics-opencv-headless`, not `ultralytics`. Otherwise pip installs `opencv-python` again.
- **Still to do:**
  - Simulate the offline timed run.
  - Test a clean `pip install -r requirements.txt` before submitting.

### D-012: How the diverse sample was picked, and the resulting subset
- **Status:** Accepted · **Date:** 2026-11-20 · Carries out D-006.
- **Decision:**
  - **Which Parquet files:** 10 files picked at random from index 50 to 1356, with seed `20261120`: `00267`, `00334`, `00486`, `00651`, `00841`, `00905`, `00919`, `00958`, `01113`, `01247`. They are stored in `data/finevideo_diverse/`.
  - **How videos are described:** `validation/select_diverse_sample.py` probes the mp4 bytes themselves with PyAV. It records codec, pixel format, resolution, rotation, frame rate, whether the timing is variable (from packet timestamps), the first timestamp and the length. The `json` metadata is recorded only for comparison.
  - **How videos are picked:** first, a greedy pass picks the videos whose attribute values are rarest, until every value seen is covered. Then the remaining places are filled with videos from the least-represented files. 35 videos are picked.
  - **Output:** `validation/subset/diverse_manifest.json` lists the chosen videos and `validation/subset/probe_diverse.jsonl` holds the probe results for all 321 videos.
- **Resulting subset:**

  | Part | Videos | Hours | Frames |
  |---|---|---|---|
  | Sequential (`train-00000`, `train-00001`) | 66 | 5.21 | 526,328 |
  | Diverse | 35 (from all 10 files, 3–4 each) | 2.38 | 240,937 |
  | **Total** | **101** | **7.59** | **767,265** |
- **What the probe found across all 387 videos:**
  - All are H.264 yuv420p with constant frame rate, starting at timestamp 0.
  - None has a rotation tag, and none has duplicate timestamps.
  - Almost all are 360p or smaller; one has a short side of 552 px.
  - Frame rates range from 15 to 30. 49 are portrait and 3 are square, all stored that way, not rotated by a tag.
  - The `json` metadata matches the probe exactly on resolution and frame rate.
- **Alternatives considered:** Picking evenly spaced files, which is no more representative than a seeded random pick. Picking videos purely at random within each file, which would miss rare cases such as frame rates below 20 or square videos.
- **Consequences:**
  - The natural data contains **none** of the "damaged or unusual" cases the brief warns about (brief §6): rotation tags, variable frame rate, timestamps not starting at 0, other codecs, corruption.
  - Those cases have to be tested with synthetic videos made on purpose (validation layer 5, D-004). How to do that is still to be discussed.
  - The resolution bucket labelled "720p" really means a short side of 481–720 px.
- **Update 2026-11-20: the subset is frozen.**
  - `validation/subset/frozen_subset.json` lists all 101 video IDs and records the size, row count and sha256 of each of the 12 source Parquet files.
  - The hashes match the ones Hugging Face lists for these files, so the downloads are intact.
  - All reference and benchmark work uses this manifest. Changing it needs a new decision.

### D-013: Synthetic fixture suite for unusual and damaged videos
- **Status:** Accepted (how the fixtures are made is still to be agreed) · **Date:** 2026-11-20 · Part of validation layer 5 (D-004).
- **Context:** The natural FineVideo data contains none of the "damaged or unusual" cases in brief §6 (see D-012). The fixture suite is kept separate from the 101-video reference.
- **Decision:** Fixtures are prioritised. **Every Critical fixture must pass before an optimisation is adopted.** High and Medium fixtures are secondary, because they cost more to build and run.

  | Test case | Priority | What must be verified |
  |---|---|---|
  | Rotation metadata 90°, 180°, 270° | Critical | Correct orientation, applied exactly once |
  | Variable frame rate | Critical | Correct second assignment from timestamps |
  | Nonzero starting timestamp | Critical | The first decoded frame belongs to second 0 |
  | Corrupted or truncated video | Critical | Completed results kept, error line written |
  | Empty or null mp4 bytes | Critical | Error line, no pipeline crash |
  | Non-video bytes | Critical | Error line, and the next video still processes |
  | Audio-only file | Critical | "No video stream" handled correctly |
  | Timestamp gap across seconds | Critical | Empty seconds written correctly |
  | One-frame or sub-second video | Critical | The last second isn't left out |
  | **Timestamps just before, exactly on and just after whole-second boundaries** | Critical | Exact assignment to a second |
  | **Video that decodes several seconds and then fails** | Critical | Brief's partial-failure rule: earlier lines kept, plus one error line |
  | Other resolutions and aspect ratios | High | Correct letterboxing |
  | 60 fps | High | Every frame processed and counted |
  | HEVC, VP9, AV1 | High | Supported files work, unsupported ones fail cleanly |
  | Missing or duplicate timestamps | High | Clearly defined fallback or error behaviour |
  | 1080p and 4K | Medium | Memory stays bounded |
  | Multiple video streams | Medium | Consistent stream selection |
- **Alternatives considered:**
  - Relying on natural data only: none of these cases occur in it (D-012).
  - Making every case mandatory: the expensive High and Medium fixtures would slow down each optimisation round.
- **Consequences:**
  - Each fixture needs an expected output that can be derived by hand or independently, not just "whatever the reference produces".
  - Fixtures will be generated in code (PyAV) so they can be reproduced.

### D-014: Every-frame reference builder design
- **Status:** Accepted, with the items marked ⚠ pending the user's confirmation · **Date:** 2026-11-20 · Carries out D-003.
- **Code:**
  - `validation/build_reference.py` stores per-frame detections, one Parquet file plus one JSON file per video, in `work/reference/frames/`.
  - `validation/reference_aggregate.py` turns them into `reference.jsonl`.
  - Both are deliberately independent of `run.py`, so a bug isn't shared between the reference and the pipeline it checks.
- **Model settings:**
  - `imgsz=640, conf=0.4, iou=0.7` are passed explicitly. Everything else uses the Ultralytics defaults: FP32, class-aware NMS, `max_det=300`, and a letterbox with `auto=True` when every frame in a batch has the same shape (the stride-padded minimum rectangle).
  - **Finding: in `ultralytics 8.4.215` the default IoU for predict is 0.5 (`PREDICT_IOU`), not the 0.7 the brief calls "the default".** Relying on defaults would therefore be wrong. `run.py` must pass `iou=0.7` too.
- **Decoding:**
  - PyAV decodes every frame in presentation order. Frames are converted to BGR with `to_ndarray("bgr24")`.
  - Rotation comes from the display matrix (`frame.rotation`, counterclockwise degrees) and is applied with `np.rot90(k=rotation // 90)`. ⚠ The direction still has to be checked against the D-013 rotation fixtures.
- **Timestamps:**
  - Each frame's time is `t = (pts - first_pts) * time_base`, computed as an exact `Fraction`, and its second is `floor(t)`. There is no floating-point error at second boundaries.
  - A frame without a timestamp ends the video with an error.
- **Aggregation (brief §5):**
  - An object is present when `2 * frames_with_object >= n_frames`.
  - `people` is the floor of the median of the per-frame person counts.
  - Lines run from second 0 to the second of the last frame.
  - On an error, completed second lines are kept and one error line is added.
- **⚠ Interpretations that need confirmation:**
  1. **No class filter during inference:** all 80 classes go through NMS, and only the 5 targets are kept afterwards. This is the pure-defaults reading. Because NMS is class-aware, the `classes=` filter only changes results when a frame has more than 300 candidate detections.
  2. **A second with zero frames** (a timestamp gap) gets `objects: []` and `people: 0`. That's our reading of "a second with nothing found"; the brief's median rule isn't defined for zero frames.
  3. **PyAV as the reference decoder.** The graders' decoder is unknown, and `cv2.VideoCapture` is the other likely choice. Layer 1 should compare the two.
- **Run:**
  - Two GPUs, one worker process each (D-007), batch size 32, decoding in a separate thread.
  - The run resumes where it left off, using a per-video `.json` file as the done marker.
  - It refuses to start if any Parquet file's sha256 differs from the frozen manifest.
- **Smoke test:** 2 videos ran at about 150 fps on one RTX 6000, with the expected number of second lines.
- **Result 2026-11-20 (reference built and checked):**
  - **Run:** 101 videos and 767,265 frames in 45.3 minutes on 2 RTX 6000s, about 282 fps in total, with 0 errors.
  - **Output:** `work/reference/reference.jsonl`, 27,388 second lines (48 MB including the per-frame files). It stays on the server and is not in git.
  - **`validation/check_reference.py`: 0 failures.** It confirmed that:
    - all 101 videos are present;
    - the decoded frame count equals the probe's packet count for every video;
    - the first frame is at t = 0 and timestamps strictly increase;
    - every second from 0 to N is present with no gaps or duplicates, and no second has zero frames;
    - only the exact object names are used.
  - **Batch versus single-frame check:** 200 random frames from 8 videos were re-run one at a time through stock `predict`. They gave the same target classes, with confidence differences below 0.0005. Batch size 32 with `auto` letterbox gives the same detections as one frame at a time.
  - **How often each object appears (share of 27,388 seconds):**

    | Object | Share |
    |---|---|
    | person | 71.9% |
    | laptop | 3.6% |
    | car | 3.1% |
    | dog | 2.7% |
    | cell phone | 1.8% |

    `people > 0` in 71.6% of seconds. The person-present/`people = 0` case from a 50% tie occurs in 91 seconds.
- **Alternatives considered:**
  - Using the stock `model.predict(source=video)` path. It decodes with cv2, gives no access to exact timestamps, and is harder to make resumable.
  - Storing only per-second aggregates. That would lose the per-frame data that validation layers 2 and 3 need.

<!--
Template for new entries (copy below this line):

### D-XXX: <Title>
- **Status:** Accepted · **Date:** YYYY-MM-DD
- **Context:** Why the decision was needed.
- **Decision:** What was chosen.
- **Alternatives considered:** Each option, with why it wasn't chosen.
- **Consequences:** Trade-offs, risks, and what this rules out or depends on.
-->
