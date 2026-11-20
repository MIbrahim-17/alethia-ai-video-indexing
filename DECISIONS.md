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
| D-015 | Baseline `run.py` and scorer design | Accepted (baseline v1 measured 2026-11-20) | 2026-11-20 |
| D-016 | Frame sampling: techniques tested and results | Experiment done (strategy chosen in D-017) | 2026-11-20 |
| D-017 | Staged sampling `staged-1>3>all-VPN` in `run.py` | Accepted, provisional (holdout passed 2026-11-20) | 2026-11-20 |
| D-018 | Untouched correctness holdout | Accepted | 2026-11-20 |

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
- **Update 2026-11-20 (Critical suite built; all 17 fixtures pass against the baseline `run.py`):**
  - **Generator:** `validation/fixtures/make_fixtures.py` writes all fixtures to one Parquet file (`work/fixtures/input/fixtures-critical.parquet`) plus a manifest.
    - **Content:** P is a real FineVideo frame with one person the model is confident about (`train-00000:10`, frame 156, confidence 0.96). B is a flat grey frame.
    - **Boundary and variable-frame-rate cases:** P and B frames alternate so that putting a frame in the wrong second changes the expected output.
    - **Encoding:** H.264, time base 1/90000.
  - **Checker:** `validation/fixtures/check_fixtures.py` derives each expected result independently of `run.py`:
    - decoding with cv2's FFmpeg build, using OpenCV's own automatic rotation;
    - timestamps from the times the fixture was built with;
    - stock single-frame `predict`;
    - grouping with `reference_aggregate.py`.

    It also checks that each fixture is valid (the container's timestamps equal the intended ones). For rotation, it compares `run.py`'s decoded pixels with cv2's.
  - **Unit tests:** `validation/test_run_unit.py` has 12 tests (layer 3) with a fake decoder and detector:
    - ties and the floor of the median;
    - gaps and exact second boundaries;
    - failure partway through, in both keep and drop modes;
    - timestamps going backwards;
    - empty bytes;
    - the rotation mapping.

    All pass.
  - **Rotation verified:** for 90°, 180° and 270°, `run.py`'s frames match cv2's automatic rotation pixel for pixel (difference 0.00 against 70 for a wrong turn), with the portrait shape. This closes the ⚠ rotation item in D-014 and D-015 C3.
  - **Lesson from the first version:** damaged fixtures made from a near-lossless encode with a single keyframe were invalid. Frame 0 held most of the bytes, so any damage destroyed every frame, even for cv2. Damaged fixtures now use CRF 23, a keyframe every 10 frames and 100 frames.
  - **Damaged-video behaviour, which now matches cv2:**
    - a file truncated after the `moov` atom gives 6 complete seconds plus one error line, frame for frame the same as cv2, with no buffered frames lost;
    - 4 kB of random bytes in the middle is hidden by the decoder (cv2 and PyAV both decode all 100 frames), so there are 10 seconds and no error line.
  - **Not built yet:** the High and Medium fixtures (other codecs, 60 fps, 1080p and 4K, missing or duplicate timestamps, multiple streams).

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
- **⚠ Rotation source refined by D-015:** read `frame.rotation` first, then fall back to the stream's `rotate` tag.
- **Alternatives considered:**
  - Using the stock `model.predict(source=video)` path. It decodes with cv2, gives no access to exact timestamps, and is harder to make resumable.
  - Storing only per-second aggregates. That would lose the per-frame data that validation layers 2 and 3 need.

### D-015: Baseline `run.py` and scorer design
- **Status:** Accepted · **Date:** 2026-11-20 · Carries out D-003, the first working version.
- **What the baseline is:** an every-frame, batched, single-process pipeline on one GPU. Batching is already an optimisation, so this is described as a batched baseline, not as "unoptimised".
- **Decisions:**

  | # | Topic | Decision | Rejected alternative |
  |---|---|---|---|
  | 1 | Finding input | `sorted(glob("*.parquet"))` in the `--input` folder only, not subfolders | Recursive search: the brief says "a folder of Parquet files". Unsorted: order would vary between runs. |
  | 2 | Unreadable Parquet file | If the footer is readable, write one valid error line per row (`file:0` … `file:N-1`). If not, report on stderr only and invent no video ID. The exit code is 0 for data problems; nonzero is only for crashes. | `file:?` (breaks the ID format). A nonzero exit (could make the whole run look failed). |
  | 3 | Partial second after a failure partway through | **Keep it:** group the frames decoded so far, write that second, then the error line. It's a setting (`--drop-partial-second`) so it can be flipped. | Dropping it (proposed by the user, then withdrawn). Every normal video already ends on a partial second that the brief requires, we can't tell whether the second was really unfinished, and the graders' reference most likely groups whatever frames it decoded. This is a judgement call. |
  | 4 | F1 averaging | Report micro, macro, per-class and sample-averaged F1. **Gate on min(micro, macro)**, with a margin above 0.95. Sample-averaged F1 is for information only, because the empty-versus-empty convention inflates it. | Gating on micro alone: hides failures on rare objects, since cell phone appears in only 1.8% of seconds. |
  | C1 | Writing | Write a second as soon as a frame from a later second arrives. A frame landing in an already-written second is an error for that video. One writer only. Flush after each video. | Buffering whole videos: uses more memory and loses work if the run is stopped at the 6-hour limit. |
  | C2 | Batching | Batches of up to 32 frames, **never mixing videos**, so all frames in a batch have the same shape and Ultralytics's `auto` letterbox matches single-frame results (checked, D-014). Batch size 1 is timed once as the naive starting point. | Batches across videos: mixed shapes switch to `auto=False` and pad to 640×640, which gives different detections. |
  | C3 | Rotation | Isolated code: read `frame.rotation`, falling back to the stream's `rotate` tag. Apply it once with `np.rot90`. It's adopted only after the D-013 rotation fixtures pass against an independent tool (cv2 automatic rotation or ffmpeg automatic rotation). The baseline isn't held up for this, because no real video is rotated (D-012). | Blocking the baseline until rotation is verified: no score effect on real data. |
  | C4 | Timestamps | `t = pts × frame.time_base − pts₀ × time_base₀` with exact fractions, falling back to the stream's time base. `frame.time` (a float) is never used. A frame without a timestamp is an error, and the D-013 fixture will define what happens then. `check_reference.py` gains a check that frame and stream time bases agree. | Stream time base only. Float times: wrong at exact second boundaries. |
  | C5 | Memory | Decode queues and batches are limited by **bytes** (about 1 GB), so large frames make smaller batches. If CUDA runs out of memory, the batch is halved and retried before the video is given up with an error line. The same byte limit goes into `build_reference.py`. | Limits by frame count: a 256-frame queue at 4K is about 6.4 GB. Writing an error line straight away: loses all of a good video's seconds. |
- **Scorer:** `validation/score.py` scores an output against `reference.jsonl`.
  - A missing second counts as wrong. Extra or duplicate lines fail the check outright.
  - It reports the brief's three metrics, the F1 variants in row 4, and breakdowns per video and per object.
- **Result 2026-11-20 (baseline v1 measured):**
  - **Accuracy:** on all 101 videos (27,388 seconds), the output is **identical to the reference**. Every F1 and both people metrics are 1.0000, with no missing, extra or duplicate lines and no errors. Fixtures: 17 of 17 pass. Unit tests: 12 of 12 pass.
  - **Timing** (GPU 0, the 2 complete sequential files, run with no options; logged as S-1 in the Speed log):

    | Measure | Value |
    |---|---|
    | Videos | 66 |
    | Frames | 526,328 |
    | Video length | 5.21 h |
    | Wall time | **3,833 s (63.9 min)** |
    | Speed | 137 fps |
    | Wall time per second of video | 0.204 s |
    | CPU | 105% (a single core) |
    | Peak memory (RSS) | 3.9 GB |
  - **Caveat:** for the first 33 minutes, the diverse accuracy run was using GPU 1 at the same time. There may have been a little CPU contention, so the numbers may be slightly pessimistic.
  - **Estimates:**
    - full 50 files (about 125 h of video): about **25.5 h** on one RTX 6000;
    - the graders' 13 h test set: about **2.7 h** on an RTX 6000 (a 4090 will be faster; to be measured, not assumed).
  - **Time-base check:** all 767,265 reference frames have `frame.time_base == stream.time_base`, so the reference stays valid with C4.
  - **Where the time goes:** decoding alone is cheap. Decoding all 767k frames took about 67 CPU-seconds in total (about 11,000 fps per core, without converting to arrays). So the baseline is limited by inference and its single-threaded preprocessing (GPU at about 90%, one CPU core).
- **Noted for optimisation later, not part of the baseline:** shrinking frames early, but only with exactly the same resize Ultralytics uses (the same size, `cv2.resize` INTER_LINEAR). It must pass validation layer 2 first.

### D-016: Frame sampling: techniques tested and results
- **Status:** Experiment done; **strategy not chosen yet** (awaiting the user) · **Date:** 2026-11-20 · Follows D-003 step 3.
- **Priority (set by the user):** accuracy and correctness first, speed second.
- **Method:** `validation/simulate_sampling.py`.
  - The reference stores every frame's detections (D-014), and batched and single-frame inference agree. So any sampling strategy's output can be computed exactly from those stored detections: select the frames, aggregate them with the brief §5 rules, and score against the every-frame reference. No GPU is needed (about 30 s for 50 strategies).
  - **Self-check:** the simulator's every-frame aggregation reproduces `reference.jsonl` exactly on all 27,388 seconds.
  - **Cost** = the share of frames that go through the model. Decoding still covers every frame, but it's cheap (D-015).
  - **Metrics:** pooled over all seconds of all 101 videos. Robustness (layer 5): the 5th percentile of a 2,000-sample bootstrap over videos (p05), the sequential and diverse parts separately, and the worst single video.
  - The simulator scores with its own code. The chosen strategy will also be run for real through `run.py` and scored with `score.py`.
- **Techniques tested** (n = frames in the second, in time order; k = frames sampled per second):

  | Name | Exact rule |
  |---|---|
  | `uniform-k` | k frames at index positions `floor((j + 0.5) · n / k)`, j = 0…k−1, in each second. All frames if n ≤ k. |
  | `time-k` | the frame nearest each time `s + (j + 0.5) / k`. The same as `uniform-k` for constant frame rate; included to check variable frame rate. |
  | `stride-N` | every Nth frame of the video (global frame index mod N = 0). A second that gets no frame uses its middle frame. |
  | `adapt-k-F` | `uniform-k`; if the second is uncertain by any criterion in F, run the model on **all** its frames. |
  | `staged-k1>k2>…-F` | `uniform-k1`; while the second is still uncertain, add the `uniform-k2` frames, then `k3`, … (`all` = every frame). The result uses every frame run so far. With k2 = 3·k1 the positions nest (1 → 3 → 9), so no inferred frame is wasted. |

  Uncertainty criteria:
  - **V:** the sampled frames disagree on whether some object is present.
  - **P:** the sampled frames disagree on the person count.
  - **C:** some sampled detection has confidence below 0.5 (0.45, 0.6 and 0.7 were also tried).
  - **N:** the second's first-stage result (objects and people) differs from the previous or next second's first-stage result. It needs one second of look-ahead and is checked at the first stage only.
- **Results** (selected rows; all rows are in `work/sampling/sim*.json` on the server). Thresholds: F1 ≥ 0.95, exact ≥ 0.85, within ±1 ≥ 0.95. "F1" is min(micro, macro) (D-015 #4).

  | Strategy | Frames run | F1 (p05) | Lowest class F1 | People exact (p05) | Within ±1 (p05) |
  |---|---|---|---|---|---|
  | all frames | 100% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
  | uniform-1 | 3.6% | 0.932 (0.890) **fail** | — | 0.920 (0.897) | 0.990 |
  | uniform-2 | 7.1% | 0.893 (0.826) **fail** | — | 0.915 | 0.993 |
  | uniform-3 | 10.7% | 0.964 (0.941) | cell phone 0.919 | 0.952 (0.936) | 0.998 |
  | uniform-5 | 17.8% | 0.975 (0.958) | cell phone 0.949 | 0.965 (0.952) | 0.999 |
  | stride-5 | 20.0% | 0.965 (0.943) | — | 0.958 (0.945) | 0.998 |
  | stride-10 | 10.0% | 0.943 (0.909) **fail** | — | 0.937 | 0.994 |
  | staged-1>3>9-VN | 8.3% | 0.981 (0.972) | cell phone 0.965 | 0.959 (0.944) | 0.998 (0.997) |
  | staged-3>9-V | 13.6% | 0.983 (0.976) | cell phone 0.969 | 0.961 (0.946) | 0.998 |
  | **staged-1>3>9>all-VN** | **15.8%** | **0.994 (0.992)** | cell phone 0.987 | **0.966 (0.953)** | **0.998 (0.997)** |
  | staged-3>9>all-V | 22.9% | 0.998 (0.997) | cell phone 0.994 | 0.969 (0.956) | 0.998 |
  | **staged-1>3>all-VPN** | **25.0%** | **0.995 (0.992)** | cell phone 0.988 | **0.994 (0.992)** | **1.000 (1.000)** |
  | staged-3>9>all-VP | 33.4% | 0.999 (0.998) | cell phone 0.995 | 0.998 (0.998) | 1.000 |
  | staged-5>15>all-VPC | 46.2% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
- **Findings:**
  1. **Even k is worse than k − 1.** A 1-of-2 split is a tie, which counts as present (brief §5: at least half), so uniform-2 has lower F1 than uniform-1, and uniform-4 than uniform-3. Only odd stage sizes are used.
  2. **Object F1 is the binding metric**, and the rarest class (cell phone, 1.8% of seconds) limits macro F1. People exact has a lot of slack (0.85 needed, 0.95 or more reached).
  3. **Escalating on person-count disagreement (P) is expensive**, because counts change between frames often. It buys people exactness, which has the most slack.
  4. **The neighbour test (N) is what makes a 1-frame first stage work.** It catches objects appearing or disappearing between seconds.
  5. **The confidence test (C) adds cost for little gain.** It can only see detections at or above 0.4, because detections below the threshold aren't stored.
  6. Sequential and diverse parts behave the same, within about 0.01.
  7. Single videos can be much worse than the pool: the worst video's people exact is 0.59 to 0.65 without P, and 0.89 with `staged-1>3>all-VPN`. The grade is pooled over about 13 h, so this matters only if the test set is a few videos.
- **Caveats:**
  - These results hold for FP32 PyTorch detections. FP16 or TensorRT change confidences slightly, so the chosen strategy must be re-validated after any inference-engine change. The errors may add up.
  - The real wall time is not proportional to frames run: decoding every frame, buffering, and more inference passes per second add cost. It has to be timed (D-002).
- **Implementation needs** (for any staged strategy): buffer one second of decoded frames (as YUV `VideoFrame`s; only frames sent to the model are converted to BGR), keep one second of look-ahead for N, and batch first-stage and escalation frames from the same video together (the shapes stay equal, D-015 C2).
- **Alternatives not tested:**
  - Scene-change or frame-difference sampling: it needs pixel work on every frame, and the neighbour test already covers changes between seconds.
  - Running the model at conf 0.25 to see detections just below 0.4: that can't be simulated from the stored reference and would need a new GPU run.

- **Update 2026-11-20:** the user chose `staged-1>3>all-VPN`, provisionally (D-017). `staged-1>3>9>all-VN` (15.8% of frames) stays documented here as the faster alternative, to consider only if VPN's measured runtime isn't enough. It isn't implemented in `run.py`.

### D-017: Staged sampling `staged-1>3>all-VPN` in `run.py`
- **Status:** Accepted, **provisional** (the user's choice, pending the holdout check in D-018) · **Date:** 2026-11-20 · Carries out D-016.
- **Decision:**
  - `run.py --sampling staged-1>3>all-VPN` is the default.
  - `--sampling all` is the every-frame baseline (D-015), kept as the fallback.
  - No other strategy is selectable.
- **Rules (exactly the simulator's `strat_staged`):**
  1. **Stage 1:** the middle frame of each second (position `n // 2`).
  2. **Escalate to stage 2** when the second has more than one frame and **N** holds: its stage-1 result (objects, people) differs from the previous or next second's stage-1 result. A second with no frames counts as (no objects, 0). The first second has no previous neighbour, and the last second has no next one.
  3. **Stage 2:** add the uniform-3 frames (positions `(2j + 1) · n // 6`, which include the middle frame). If frames are still left out and **V** (the frames disagree on some object) or **P** (they disagree on the person count) holds, go to stage 3.
  4. **Stage 3:** every frame of the second.
  5. **Aggregation:** the brief §5 rules over every frame run for that second.
- **Implementation:**
  - Decoded frames are kept as YUV `VideoFrame`s. Only frames sent to the model are converted to BGR and rotated (`frame_to_bgr`).
  - Up to 32 seconds or about 1 GB of frames are buffered per chunk. The last second of each chunk waits for its next neighbour.
  - Each stage is batched over every buffered second (batches of at most 32 frames or 1 GB, never mixing videos). A frame is never run twice.
  - The error handling is the same as the baseline's: a decode failure or a timestamp going back keeps every second decoded so far (the partial second too, unless `--drop-partial-second`), then writes one error line.
- **Changes to shared code (the baseline path is unaffected):**
  - Decoding is split into `iter_raw_frames` plus `frame_to_bgr`; `iter_frames` is rebuilt on top of them and does the same work in the same order.
  - `Detector._predict` now sends only same-shape runs to one predict call. This makes the D-015 C2 rule hold even if a video changes resolution partway; no real video does.
  - It also counts predict calls and inferred frames.
- **Validation (2026-11-20):**
  - **Layer 3 (unit tests):** 18 of 18 pass, 6 of them new. On 60 random synthetic videos (frame rates 1–60, gaps, videos up to 4 chunks long), the output and the set of frames sent to the model equal the simulator's exactly, and no frame is run twice. `uniform_positions` equals the simulator's float rule for n < 400 and k ∈ {1, 3, 5, 9, 15}.
  - **Real run on all 101 videos, GPU 0:** the output equals the simulator's prediction on **all 27,388 seconds (0 differences)**. The inferred-frame count equals the simulator's for **every one of the 101 videos** (191,848 in total, 25.0%). No differences were found, so there was nothing to investigate.
  - **`score.py` (independent of the simulator):** F1 micro 0.9978, macro 0.9947, gate 0.9947. Lowest class: cell phone 0.9876. People exact 0.9940, within ±1 0.9999. Complete, PASS. The worst video's people exact is 0.889 (`train-00001:19`).
  - **Fixtures:** 17 of 17 pass with `--sampling staged-1>3>all-VPN` (the oracle applies the simulator's selection to cv2-decoded, single-frame-predicted frames) and 17 of 17 with `--sampling all`.
  - **The fallback still equals the baseline:** `--sampling all` on 6 diverse videos (37,741 frames, 1,292 seconds) is byte-for-byte identical to the reference, at 136.5 fps (the baseline speed).
  - **Timing:** S-3 in the Speed log. The 66 sequential videos took **1,049 s against 3,833 s, 3.65× faster.**
- **Observations for the next optimisations:**
  - Only 24% of frames are run, yet the speed-up is 3.65×, not 4.2×. Frames run per second of wall time fell from 137 to 120. Likely causes: smaller and partial batches (4,494 predict calls, about 28 frames per call), decoding every frame, and Python work per second, all on one CPU core (CPU 105%).
  - Peak memory rose from 3.9 to 5.4 GB because of frame buffering.
- **Alternatives considered:**
  - `staged-1>3>9>all-VN` (faster, people exact about 0.966); documented only (D-016 update).
  - Decoding each video twice instead of buffering frames: simpler memory use, but double decoding.
- **Update 2026-11-20:** it passes the untouched holdout (D-018): F1 gate 0.9947, people exact 0.9950, within ±1 0.9999, identical to the simulator. It's still provisional, as the user asked, and stays the default until the user confirms it or a later change (for example the inference engine) forces a re-check.

### D-018: Untouched correctness holdout
- **Status:** Accepted · **Date:** 2026-11-20 · Validation layer 5 (D-004).
- **Context:** The 101-video subset was used to choose the sampling strategy, so its scores are optimistic. An untouched set is needed to confirm a chosen configuration.
- **Decision:**
  - `validation/holdout/select_holdout.py` with seed 20261121. It draws **8 files uniformly** from `train-00050` … `train-01356` (leaving out the 50 graded files and every file in the 101-video subset): 00201, 00218, 00409, 00628, 00745, 00767, 01137 and 01307.
  - Then it draws **6 rows uniformly per file** (seed + 1), for **48 videos**.
  - This is a plain random sample, unlike the rarity-weighted diverse sample (D-012), so it stands for FineVideo as it is.
  - The sample is frozen with sha256 values in `validation/holdout/holdout_subset.json`.
  - **Rule:** the holdout only confirms or rejects a configuration that has already been chosen. Nothing is tuned on it, and every result is recorded whatever it is.
- **Reference:**
  - Built with the same `build_reference.py` on both GPUs: 48 videos, 403,553 frames, 14,341 seconds (3.98 h of video), in 24.6 min.
  - `check_reference.py`: 0 failures. Frame counts match the probe (`probe_holdout.py`, the same `probe_video` as D-012), time bases agree on all 403,553 frames, and 400 frames re-run one at a time match the batched results.
  - The reference stays on the server (`work/holdout_ref/`).
- **Result for `staged-1>3>all-VPN` (D-017), GPU 0:**
  - F1 micro 0.9979, macro 0.9947, **gate 0.9947**. Lowest class: cell phone 0.9847.
  - **People exact 0.9950, within ±1 0.9999.** Complete, PASS.
  - The worst video's people exact is 0.978.
  - 96,149 frames run (23.8%), 3,445 predict calls.
  - The output equals the simulator's prediction on all 14,341 seconds, and the frame count matches for all 48 videos.
  - Timing: S-4.
- **Alternatives considered:**
  - Reusing the diverse sample: it was used for the choice, so it isn't untouched.
  - A holdout from the 50 graded files: those are the final output, and the brief's test set is hidden, so a separate set is cleaner.
  - Feature-weighted selection: it's biased toward rare cases. The diverse sample already covers those.
- **Consequences:** about 3.9 GB of disk (46 GB free afterwards).
---

## Speed log

Every timed run, in order, so each optimisation can be compared with the one before it. A run only counts toward a decision if it is on GPU 0 (D-007) and its output passes validation (D-004). "s per video-s" is wall time divided by video length. The estimates scale that ratio linearly and don't include model loading.

| # | Date | Version (decision) | Run | GPU | Videos / frames / video length | Wall | Decoded fps | s per video-s | Est. full 50 files (≈125 h) | Est. 13 h test set | Accuracy vs reference | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-1 | 2026-11-20 | Baseline v1, every frame, batch 32, FP32 PyTorch (D-015) | sequential subset (`train-00000`, `train-00001`) | 0 (RTX 6000) | 66 / 526,328 / 5.21 h | 3,833 s (63.9 min) | 137.3 | 0.204 | ≈25.5 h | ≈2.7 h | identical (all metrics 1.0000) | CPU 105% (one core), peak RSS 3.9 GB, GPU ≈90%. GPU 1 busy with S-2 for the first 33 min. **The timing figure of record for the baseline.** |
| S-2 | 2026-11-20 | Baseline v1 (D-015) | diverse subset (10 files) | 1 (RTX 6000, ≈10.7 GB held by ollama) | 35 / 240,937 / 2.38 h | 1,988 s (33.1 min) | 121.2 | 0.232 | — | — | identical (all metrics 1.0000) | Run for accuracy only. GPU 1 is not the benchmark GPU, so this timing isn't compared with others. |
| S-3 | 2026-11-20 | Staged sampling `staged-1>3>all-VPN` (D-017) | sequential subset (`train-00000`, `train-00001`) | 0 (RTX 6000), alone | 66 / 526,328 / 5.21 h | **1,049 s (17.5 min)**; 17:30 for the whole process | 501.6 (frames run per second of wall time: 120.3) | 0.0559 | ≈7.0 h | ≈0.73 h (44 min) | F1 gate 0.9947, exact 0.9940, within ±1 0.9999 on all 101 videos; identical to the simulator | 126,201 frames run (24.0%), 4,494 predict calls, CPU 105%, peak RSS 5.43 GB. **3.65× faster than S-1.** |
| S-3b | 2026-11-20 | Same as S-3 | diverse subset (10 files) | 0 (RTX 6000) | 35 / 240,937 / 2.38 h | 541 s (9.0 min) | 445.1 | 0.0632 | — | — | (pooled with S-3 above) | 65,647 frames run (27.2%), 2,335 predict calls, peak RSS 5.51 GB. The holdout download (network only) overlapped this run, so it's for information only. |
| S-4 | 2026-11-20 | `staged-1>3>all-VPN` (D-017) | holdout (D-018) | 0 (RTX 6000), alone | 48 / 403,553 / 3.98 h | 827 s (13.8 min) | 487.7 | 0.0577 | ≈7.2 h | ≈0.75 h | holdout: F1 gate 0.9947, exact 0.9950, within ±1 0.9999 | 96,149 frames run (23.8%), 3,445 predict calls, peak RSS 5.58 GB. Agrees with S-3 (0.0559 s per video-s). |

<!--
Template for new entries (copy below this line):

### D-XXX: <Title>
- **Status:** Accepted · **Date:** YYYY-MM-DD
- **Context:** Why the decision was needed.
- **Decision:** What was chosen.
- **Alternatives considered:** Each option, with why it wasn't chosen.
- **Consequences:** Trade-offs, risks, and what this rules out or depends on.
-->
