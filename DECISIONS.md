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

<!--
Template for new entries (copy below this line):

### D-XXX: <Title>
- **Status:** Accepted · **Date:** YYYY-MM-DD
- **Context:** Why the decision was needed.
- **Decision:** What was chosen.
- **Alternatives considered:** Each option, with why it wasn't chosen.
- **Consequences:** Trade-offs, risks, and what this rules out or depends on.
-->
