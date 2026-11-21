# Documentation audit

Audit of `APPROACH.md`, `README.md`, `DECISIONS.md`, `HARDWARE.md` and `ai_chats/README.md` against the code, the tests and the raw benchmark artifacts. Audit date: 2026-11-21, at commit `e462731`.

- `WRITEUP.md` was not audited (brief §8), and it doesn't exist yet.
- `validation/` contains no other Markdown files.
- No documentation, code, logs or git history were changed during this pass.

**Evidence used:**
- **Code, read statically:** `run.py`, `prep.py`, `requirements.txt` and the `validation/` tools.
- **Ultralytics 8.4.215 source** in the server's venv.
- **Raw logs** on the development server (`~/alethia/work/*/`): every `DONE` line, `/usr/bin/time -v` output, `profile/p0.json`, `sampling/sim.json` and `cleantest/pip_attempt3.log`.
- **The chat transcript**, for test output that was only printed to the terminal.
- **Checks re-run for this audit, all CPU-only:**
  - `validation/test_run_unit.py`: 23 of 23 pass;
  - `validation/check_full_results.py` on the frozen results for files 00000 and 00001: `COMPLETE=True`;
  - set-disjointness checks on the subset manifests;
  - `torch.cuda.get_arch_list()`;
  - `free -b` and `lscpu`.
- **Not re-run:** no GPU job, dataset run, download or engine build was performed.

**Status key:**

| Status | Meaning |
|---|---|
| VERIFIED | Directly supported by executable code and/or reproducible evidence. |
| PARTIALLY VERIFIED | Supported only under stated conditions. |
| INACCURATE | Contradicted by the implementation or evidence. |
| OUTDATED | Was true historically but no longer describes the current version. |
| UNVERIFIABLE | The available evidence is insufficient to establish it. |

---

## A. Executive summary

| | Count |
|---|---|
| Substantive claims checked (register in section H) | **112** |
| VERIFIED | **83** |
| PARTIALLY VERIFIED | **18** |
| INACCURATE | **6** |
| OUTDATED | **3** |
| UNVERIFIABLE | **2** |
| Findings raised (section B) | 27: **0 Critical, 1 High, 5 Medium, 21 Low** |

**Overall assessment:** the documentation is reliable on what the code does and on the measured numbers.

- **Implementation:** the sampling rule, aggregation, timestamps, model settings, fallback, OOM handling and threading all match the code exactly.
- **Measurements:** every wall time, frame count and accuracy figure checked against a raw log or result file matches, and every speed-up ratio recalculates correctly.

**The problems are of three kinds:**
1. **Deployment claims are stated more strongly than the evidence supports.**
   - APPROACH.md describes the clean-environment test without saying that the final install was done with uv from a cache, not with pip.
   - Offline operation was tested only at the Python socket level.
   - Compatibility with the graders' driver and their RTX 4090 is an expectation, not something tested.
2. **Memory figures use inconsistent units.** `/usr/bin/time` reports KiB, but the docs treat the values as kB and call the result GB.
3. **Some details are left out or stale.** These include chunk size, queue accounting and error-path behaviour, plus stale statuses in DECISIONS.md.

There are **no Critical findings**, **1 High**, **5 Medium**, and the rest are Low.

---

## B. Findings

| ID | Severity | Document and lines | Claim | Actual behaviour and evidence | Status | Proposed correction |
|---|---|---|---|---|---|---|
| F-01 | **High** | APPROACH.md 293–296 (section 8, "Clean-environment test") | "Setup: a fresh venv and a fresh copy of the repository, then `prep.py`, then offline runs…", presented as a clean-environment test of the deployment. | D-027 and `cleantest/pip_attempt{1,2,3}.log`: **pip never finished an install.** Attempt 1 timed out. Attempt 2 failed resolution and led to the URL pins. Attempt 3 resolved all 67 pinned packages, but was stopped during the final downloads because the route from this server to PyPI ran at about 39 kB/s. The venv that was tested came from `uv pip install` using uv's cache, plus two wheels from a PyPI mirror whose sha256 matches PyPI's. The copy of the repo was `git archive HEAD`, not a clone. | PARTIALLY VERIFIED | See G-1. |
| F-02 | Medium | README.md 7 | "Tested with Python 3.11 and an NVIDIA driver with CUDA 12 support." | Tested only on driver 595.99.02, which reports CUDA 13.2, under Ubuntu 26.04 (HARDWARE.md, D-001). No run used a driver that supports only CUDA 12, or Ubuntu 22.04. The torch build has `sm_86` code but no `sm_89` (`get_arch_list()` gives `sm_50…sm_86, sm_90`). The RTX 4090 is `sm_89`, so the PyTorch fallback relies on the `sm_86` code running there. That is expected, but untested. | PARTIALLY VERIFIED | See G-2. |
| F-03 | Medium | README.md 82; APPROACH.md 297 | "Expect the same size of difference between `results.jsonl.gz` and a run on the grading machine" / "A run on the grading machine will differ from `results.jsonl.gz` by about this much." | The only evidence is a second engine built **on the same GPU model** (Quadro RTX 6000), which changed 49 of 41,729 seconds. The RTX 4090 is a different architecture, with different kernels and FP16 numerics. No measurement exists for it. | UNVERIFIABLE (a prediction) | See G-3. |
| F-04 | Medium | APPROACH.md 297 | "With the original engine, the clean environment reproduced S-10 and `results.jsonl.gz` byte for byte." | The check ran on the **66-video sequential set (files 00000–00001) only**: `out/seq_oldengine.jsonl` is byte-identical to `s10/seq_t1.jsonl` and to those lines of `results.jsonl.gz`. The diverse and holdout sets weren't re-run with the old engine. D-027 states this correctly. | PARTIALLY VERIFIED | Add "on the 66-video sequential set (files 00000–00001)". |
| F-05 | Medium | README.md 37; APPROACH.md 292 | "Nothing needs the network: `run.py` sets `YOLO_OFFLINE=1` and turns off Ultralytics' auto-install." | `run.py:26-27` sets both variables, and Ultralytics' `is_online()` returns False when `YOLO_OFFLINE` is set (`ultralytics/utils/__init__.py:1129`). The test is `validation/offline_guard.py`, which blocks **Python-level** DNS and socket calls only, and showed 0 attempts in 5 runs. Native code wasn't blocked, and no run was made in a network-isolated namespace, which the server doesn't allow (D-025). | PARTIALLY VERIFIED | See G-4. |
| F-06 | Medium | README.md 54 | "…the run carries on. For a video, it writes an `{"video": ..., "error": ...}` line and keeps the seconds it had finished." | True for lines already written. Three behaviours aren't described: (a) after a **decode** failure, every buffered second is still processed and written, **including the unfinished last second** (`keep_partial` defaults to True, `run.py:596-600`); (b) after a failure **during inference** (a single-frame OOM, `run.py:286-287`, or an unexpected exception inside `StagedVideo.process`), the decoded seconds in the current chunk that haven't been written yet, up to 33 seconds, are dropped (`run.py:601-606` only flushes the accumulator); (c) a Parquet file whose footer reads but whose rows don't gets an error line **for each remaining row** (`run.py:841-846`, `869-875`). Error messages are cut to 200 characters (`run.py:56`). | PARTIALLY VERIFIED | See G-5. |
| F-07 | Low | README.md 39 | "It's slower, but the results agree with the engine's to within the accuracy figures below." | The figures below are TensorRT's. The PyTorch FP16 path's own scores are in D-022 (dev 0.9938 / 0.9928 / 0.9999; holdout 0.9947 / 0.9939 / 0.9999), from `work/s6` and `work/fp16`. | INACCURATE | "…carries on with PyTorch FP16. It's slower, and it was validated separately (D-022: dev F1 0.9938, people exact 0.9928)." |
| F-08 | Low | README.md 66; DECISIONS.md D-017 "Up to 32 seconds" | Chunks of 32 seconds. | `StagedVideo.add_second` processes when `len(self.pending) > CHUNK_SECONDS`, so at **33** pending seconds (`run.py:504`). The class docstring says CHUNK_SECONDS + 1 (`run.py:434`). | INACCURATE (off by one) | "chunks of up to 33 seconds (32 plus one second of look-ahead) or 1 GB". |
| F-09 | Low | README.md 62; APPROACH.md 213 (diagram) | "A bounded queue of 512 MB of decoded frames" / "at most 512 MB of frames". | `_ByteQueue` counts the **estimated YUV 4:2:0 size**, width × height × 1.5 (`frame_bytes`, `run.py:138-145`), not real memory. It always lets one item in, even if that item is over the limit (`run.py:636`). The measured queue peak was 512 MB (full run DONE line). | PARTIALLY VERIFIED | "a queue limited to about 512 MB of decoded frames (by their YUV size)". |
| F-10 | Low | README.md 75 | Column header "F1 (macro gate)". | The gate is `min(micro, macro)` (`score.py`, `f1_gate`). The values shown are macro because macro is the lower of the two. | INACCURATE (label) | "F1 gate (min of micro and macro)". |
| F-11 | Low | README.md 29; APPROACH.md 281 | "pulls about 9 GB of wheels". | `pip_attempt3.log`: 61 wheels listed for download, 8.23 GB in total, including every large wheel. The venv installed from them is **14 GB** on disk, and the README doesn't state disk needs. | PARTIALLY VERIFIED | "pulls about 8 GB of wheels (about 14 GB once installed)". |
| F-12 | Low | HARDWARE.md 9 | "RAM: 62 GiB, as reported by `free`". | `free -b`: 65,823,715,328 bytes = **61.3 GiB** (65.8 GB). `free -g` shows 61. D-001 says 61 GB. | INACCURATE | "64 GB installed (61.3 GiB usable, as reported by `free`)", or just "61 GiB usable". |
| F-13 | Low | HARDWARE.md 21; APPROACH.md 245; DECISIONS.md S-log and D-022, D-023, D-025, D-026 | Peak RAM "6.2 GB", "6.16 GB", "7.9 GB", "7.2 GB" and others. | `/usr/bin/time -v` reports **KiB**. For example, the full run's 6,159,932 KiB is 6.31 GB, or 5.87 GiB; the TRT timing run's 6,234,020 KiB is 6.38 GB / 5.95 GiB; the PyTorch control's 7,924,368 KiB is 8.11 GB / 7.56 GiB. The docs divide by 10⁶ and call the result GB, which understates the decimal-GB value by 2.4%. The comparisons between versions aren't affected. | INACCURATE (units) | State RSS in GiB: full run 5.9 GiB, TRT 5.9 GiB, PyTorch FP16 threaded 7.6 GiB. Or multiply by 1.024 for GB. |
| F-14 | Low | HARDWARE.md 21 | "about 5.8 GB of GPU memory" (as a peak). | Taken from `nvidia-smi memory.used` snapshots every 10 minutes during the run: 5,843 MiB = 6.13 GB = 5.71 GiB. That isn't a peak measurement. `test_memory.py` part C separately saw a steady 5,842 MB. | PARTIALLY VERIFIED | "about 5.7 GiB of GPU memory (5,843 MiB in `nvidia-smi` snapshots every 10 minutes)". |
| F-15 | Low | APPROACH.md 245; DECISIONS.md D-025 | "TensorRT reserves 4.6 GB of GPU memory up front, sized for its largest profile shape." | The TensorRT log line (in the transcript) reads `TensorRT-managed allocation in IExecutionContext creation: … GPU +4584, now: … GPU 4642 (MiB)`, which is 4.5 GiB, or 4.8–4.9 GB. "Sized for its largest profile shape" is an inference: the profile's maximum is 32×3×1280×1280 (`ultralytics/utils/export/engine.py:366-371`), but no measurement ties the allocation to it. | PARTIALLY VERIFIED | "TensorRT allocates about 4.5 GiB of GPU memory when the engine loads (4,584 MiB in its log)". |
| F-16 | Low | APPROACH.md 243 | "TensorRT gives the same result whether a frame runs alone or in a batch." | D-025 reports confidence difference 0.0000 and 0 mismatches for batched against single-frame runs, on a sample whose size isn't recorded. The strongest evidence is indirect: the real runs equal the simulator, which was built from separate every-frame batches, on all 191,162 + 95,621 frames run. | PARTIALLY VERIFIED | "In our tests, TensorRT gave the same result whether a frame ran alone or in a batch (a sample in D-025; and the real runs matched the simulation exactly)." |
| F-17 | Low | APPROACH.md 57 | Holdout: "final confirmation only; nothing is tuned on it". | No parameter was tuned on it, but every later experiment (FP16, TensorRT) used it as a pass/fail gate (D-020 rule 1). The sets are disjoint: 0 shared videos, and holdout files 00201, 00218, 00409, 00628, 00745, 00767, 01137 and 01307 are in neither the dev set nor the graded 0–49. It was created after the sampling rule was chosen provisionally (D-017 → D-018). | PARTIALLY VERIFIED | "used only to confirm or reject configurations already chosen; nothing is tuned on it." |
| F-18 | Low | APPROACH.md 120–128 (sampling diagram) | Stage 3 runs whenever V or P holds after stage 2. | Stage 3 also requires that frames remain unrun: `sec.unsure = len(sec.res) < len(sec.frames) and _unsure_vp(...)` (`run.py:549`). For a second with n ≤ 3, stage 2 already covers every frame (`uniform_positions`, `run.py:366-367`). The outcome is the same, since the second is complete either way. | PARTIALLY VERIFIED | Edge label: `"yes, and frames not yet run"`. |
| F-19 | Low | APPROACH.md 90 | "about 67 CPU-seconds for all 767k dev frames". | It comes from the D-015 text. No stored artifact was found. | UNVERIFIABLE | Say "(D-015, measured during development)" or remove it. |
| F-20 | Low | APPROACH.md 193 | "FP16 moves half the data and uses the GPU's tensor cores." | This is an explanation, not a measurement. The measurement is the forward pass running 1.84× faster (256.0 s → 139 s, `profile/p0.json`, `p1_fp16.json`). | PARTIALLY VERIFIED (explanation) | "…which is why it is faster (the likely reason; the measured effect is a 1.84× faster forward pass)". |
| F-21 | Low | DECISIONS.md 33–60 | The decision-log table. | It ends at D-025. The D-026 and D-027 rows are missing. | OUTDATED | Add the D-026 and D-027 rows. |
| F-22 | Low | DECISIONS.md 550 (D-020 status), 191 (D-011 "Still to do"), 49 (D-014 status), 833 (D-027) | "except the ⚠ item"; "Still to do: simulate the offline run; test a clean pip install"; "3 items pending confirmation"; "The GitHub repo isn't publicly cloneable yet." | The 5% threshold was approved (D-020 update). The offline and clean-install checks were done (D-025, D-027). The D-014 items were never formally confirmed. The repo is now public, and an anonymous clone worked. | OUTDATED | Add dated updates (never rewrite): "Update 2026-11-21: resolved in D-025/D-027", "the repo is public". |
| F-23 | Low | README.md 94, 98 | Layout lists `WRITEUP.md`; `tools/` is "helper scripts for syncing". | `WRITEUP.md` isn't in the repo yet, though it's expected before submission. `tools/` also holds `export_chats.py`. | OUTDATED / incomplete | Add "and exporting the chat history" to the tools row. |
| F-24 | Low | HARDWARE.md 8 | "24 cores (8 performance and 16 efficiency), 32 threads". | `lscpu`: 24 cores, 2 threads per core, `nproc` 32. The 8 + 16 split isn't in `lscpu`'s output; it is the published spec of the i9-14900K. | PARTIALLY VERIFIED | Keep it; optionally add "(per Intel's specification)". |
| F-25 | Low | HARDWARE.md 22 | "nothing else was running on GPU 0 during the run". | Supported by `nvidia-smi` snapshots every 10 minutes: one process's memory only, 5,843 MiB constant. That isn't continuous monitoring. | PARTIALLY VERIFIED | "no other job was started on GPU 0 during the run (checked every 10 minutes)". |
| F-26 | Low | APPROACH.md 88 | "Estimate for the full 50 files: about 25.5 h." | 0.204 s per video-second × 125 h = 25.5 h. The 50 files are actually 128.3 h (461,757 s), which gives about 26.2 h. It's correctly labelled an estimate. | PARTIALLY VERIFIED | "about 25.5–26 h (estimated)". |
| F-27 | Low | DECISIONS.md D-022, Speed log S-6 | FP16 is −37.4% against FP32. | The FP16 runs (S-6) are compared with FP32 runs from **separate sessions** (S-3, S-5), not interleaved, unlike D-023 and D-025. The same input, GPU 0 and an otherwise idle GPU are documented. The gap (1,049–1,054 s against 658 s) is about 90 times the measured noise. | PARTIALLY VERIFIED | Note "not interleaved; FP32 runs from S-3/S-5". |

---

## C. Implementation trace (current code, `run.py`)

```
python run.py --input DIR --output FILE
└─ main()                                                    run.py:899-958
   ├─ files = sorted(Path(input).glob("*.parquet"))          :928   (not recursive)
   ├─ --engine auto → Path(--weights).with_suffix(".engine") :930-936
   │    = weights/yolo11l.engine (prep.py ENGINE = WEIGHTS.with_suffix(".engine"), prep.py:31)
   │    missing → WARNING on stderr, engine=None
   ├─ Detector(weights, half=True, engine)                   :191-219
   │    engine: YOLO(engine) + warm-up predict → backend "tensorrt"
   │    any exception → WARNING, YOLO(yolo11l.pt) → backend "pytorch" (FP16 via quantize=16, :233)
   ├─ pipeline: threaded unless --sampling all               :940-941
   └─ run_threaded()                                         :850-894
      ├─ FrameProducer thread (decoder)                      :662-747
      │    pq.ParquetFile(path).iter_batches(batch_size=1, columns=["mp4"])   :706-718
      │    iter_raw_frames(mp4)                              :148-180
      │      av.open → streams.video[0] → container.decode (PyAV default threading)
      │      t = pts × time_base as Fraction; t_rel = t − t of first decoded frame
      │      rotation_k: frame.rotation, else stream "rotate" tag   :115-125
      │    → chunks of 32 frames → _ByteQueue(512 MiB, by w·h·1.5)  :622-659, :724-747
      └─ main thread, per "video" event → process_video_staged(frames=_QueuedFrames)  :567-608
           s = floor(t_rel); s < 0 or s < current second → VideoError        :589-592
           StagedVideo.add_frame → _Second.frames (YUV VideoFrames)           :449-463
             second > 512 MiB → overflow: run every frame as decoded          :459-472
           add_second: insert empty seconds for gaps; process at >32 pending seconds or ≥1 GiB  :492-505
           process()                                                          :511-564
             stage 1: uniform_positions(n,1) = [n//2]                         :516-524, :361-368
             N: stage1 ≠ previous or next second's stage1 (empty second → (no objects, 0)),
                only if frames remain                                         :527-536
             stage 2: uniform_positions(n,3) = {(2j+1)·n//6}, not yet run     :539-546
             stage 3 (k=0): every frame, if frames remain and V or P          :547-549
             write: SecondAccumulator.add over run frames, sorted by position :551-554
             release frames and results                                       :557-561
           _run_positions: frame_to_bgr (only frames sent to the model), batches ≤32 frames or 1 GiB  :412-427
           Detector.detect: same-shape runs, batch_cap per shape, OOM halving outside except  :257-301
             _predict: model.predict(imgsz=640, conf=0.4, iou=0.7)           :229-239, :31
             _summarise: 5 target classes, person count                       :241-249
           SecondAccumulator: present if 2·count ≥ n; people = floor(median); gap seconds → [] / 0  :66-110
           Writer: {"video","second","objects","people"} or {"video","error"} (msg ≤ 200 chars)  :43-63
└─ DONE summary line on stderr; exit 0 for data problems             :948-958
```

Specific checks against the documentation (section A–C of the brief's checklist):

| Item | Finding |
|---|---|
| Parquet read | `iter_batches(batch_size=1, columns=["mp4"])`. VERIFIED. |
| PyAV configuration | The default thread type is used; no thread options are set (they were removed in D-024). VERIFIED. |
| Decode overlaps inference | A separate `threading.Thread` puts frames on the queue while the main thread runs the model. PyAV's release of the GIL is a library property, not checked here; the measured CPU use of 134% against 108% (S-7 logs) is consistent with it. VERIFIED (behaviour), PARTIALLY VERIFIED (mechanism). |
| BGR only when needed | On the staged path, yes: `frame_to_bgr` is only called in `_run_positions`. The `--sampling all` path converts every frame, as documented. VERIFIED. |
| Model settings | `imgsz=640, conf=0.4, iou=0.7` are passed explicitly. NMS is class-aware by Ultralytics' default (`agnostic` isn't set), `max_det` is the default 300, and there is no class filter. Ultralytics' `PREDICT_IOU = 0.5` (`utils/__init__.py:977`) confirms that IoU must be passed. VERIFIED. |
| Letterbox for the engine | Ultralytics uses `auto = same_shapes and (format == "pt" or dynamic)` (`engine/predictor.py:236-242`), so the dynamic engine gets the same minimum-rectangle letterbox as PyTorch. VERIFIED. |
| Engine profile | min batch 1, opt 32×3×640×640, max 32×3×1280×1280 (`utils/export/engine.py:366-371`). VERIFIED. |
| TensorRT is the default | Only if `weights/yolo11l.engine` exists (`--engine auto`). PARTIALLY VERIFIED, under the stated condition; the docs say so. |
| Batch limit | 32 frames, or 1 GiB of BGR, never mixing videos (`_run_positions` is per video). VERIFIED. |
| Changing frame shapes | `detect` splits into runs of the same shape, so a resolution change midway is handled. VERIFIED. |
| OOM | Retry outside `except`; predictor state dropped; `gc.collect`; `empty_cache`; per-shape halving that persists; a single frame gives VideoError; RuntimeError "out of memory" counts as OOM. VERIFIED. |
| No frame run twice | Each stage filters out positions already run (`p not in sec.res`). The second kept back for look-ahead keeps its `stage1` and isn't re-run. VERIFIED (and `test_staged_matches_simulator_exactly`). |
| Video and second boundaries | One `StagedVideo` per video, batches per video, and seconds by `floor`. VERIFIED. |
| Aggregation | `2*count >= n`; `math.floor(statistics.median(...))`, where `statistics.median` averages the two middle values and the floor is taken afterwards. VERIFIED (and `test_median_halfway_rounds_down`). |
| Empty seconds | Written as `[]` / 0 (`run.py:99-100`). Leading gaps can't occur, because second 0 holds the first frame. VERIFIED. |
| EOF / partial last second | Written (`finish(keep_partial=True)`). VERIFIED (`test_last_partial_second_written`). |

---

## D. Benchmark verification

All wall times come from the `DONE` lines in `~/alethia/work/*/`, with `/usr/bin/time` elapsed in brackets where it was recorded.

| Claim | Documented | Raw evidence | Recalculated | Status |
|---|---|---|---|---|
| Baseline S-1 | 3,833 s | `baseline/seq.log` wall=3833.3 s (1:03:53) | — | VERIFIED |
| VPN FP32 S-3 / S-5 | 1,049 s / 1,053.7 s | `s3/seq.log` 1049.2 s; `s5/seq.log` 1053.7 s | 3,833.3 / 1,049.2 = **3.65×** | VERIFIED |
| FP16 S-6 | 658.3 / 658.4 s | `s6/seq_a.log` 658.3, `seq_b.log` 658.4 | (1,051.45 − 658.35) / 1,051.45 = **−37.4%**; 3,833.3 / 658.3 = **5.82×** | VERIFIED (see F-27) |
| Threaded S-7 | 533.8 / 534.2 s; controls 657.5 / 659.2 s | `s7/seq_t1` 533.8, `seq_t2` 534.2, `seq_c1` 657.5, `seq_c2` 659.2 | −18.7% to −19.0% against the controls; 3,833.3 / 533.8 = **7.18×** | VERIFIED |
| Decoder threading S-8 | 191.9–193.8 s; controls 192.9 / 194.2 s | `s8/scr_*.log` match exactly | maximum deviation from the control mean of 193.55 s is 0.85%. D-024 says ±0.6%; FRAME 4 is −0.85%. | PARTIALLY VERIFIED (±0.85%, still far from 5%) |
| TRT screening S-9 | 103.1 / 103.1 s; controls 193.1 / 193.7 s | `s9/*.log` match | −46.7% | VERIFIED |
| TRT S-10 | 310.2 / 310.1 s; controls 534.6 / 534.9 s | `s10/seq_t1`, `seq_t2`, `seq_c1`, `seq_c2` match | −42.0%; 1.72×; 3,833.3 / 310.2 = **12.36× ≈ 12.4×** | VERIFIED |
| Full run S-11 | 7,058 s; 1 h 57 min 38 s | `full/run.err` wall=7058.2 s; elapsed 1:57:38 | — | VERIFIED |
| Full-run counts | 50 files, 1,650 videos, 12,981,615 frames, 2,899,828 run (22.3%), 104,278 calls, 461,757 seconds, 0 errors | same `DONE` line | 2,899,828 / 12,981,615 = 22.34% | VERIFIED |
| 128 h of video | "about 128 h" | 461,757 seconds of lines | 128.3 h | VERIFIED |
| Frames per second of wall time | S-1 137.3, S-10 1,696.7, S-11 1,839.2 | frames / wall | 137.3, 1,696.7, 1,839.2 | VERIFIED |
| 13 h estimate | about 12 min (APPROACH), about 13 min (S-10 row) | rate × 46,800 s | S-11 rate gives 715 s = 11.9 min; S-10 rate gives 774 s = 12.9 min | VERIFIED (estimate, labelled) |
| Profile D-021 | 338.4 s; forward 256.0 s (75.6%); decode 31.8 s; convert 26.3 s; pre 10.6 s; post 9.0 s | `profile/p0.json` identical | 256.0 / 338.4 = 75.6% | VERIFIED |
| Decoder-only and model-only throughput | SLICE 3,912, FRAME 18,988 fps; batch 32 at 169 img/s | `p0.json` decode_only_fps, model_only_ips | 18,988 / 3,912 = 4.85× ("about 4.9×") | VERIFIED |
| Peak RSS | full run 6.16 GB; TRT 6.23 GB; PyTorch threaded 7.9 GB | 6,159,932 / 6,234,020 / 7,924,368 KiB | 5.87 / 5.95 / 7.56 GiB (6.31 / 6.38 / 8.11 GB) | INACCURATE units (F-13) |
| GPU memory (TRT) | 5,842 MB steady; 4.6 GB reserved | transcript: test output "5842 MB … peak 5842 MB"; TRT log "GPU +4584 … 4642 (MiB)" | 4,584 MiB = 4.48 GiB | VERIFIED / PARTIALLY (F-15) |
| Strategies simulated | about 50 | `sampling/sim.json` has 50 entries | — | VERIFIED |

**Comparability:**
- **S-7 and S-10** were interleaved with control runs on the same 66 videos, on GPU 0, with nothing else on it, as the logs show.
- **S-6** wasn't interleaved; see F-27.
- **S-3b** (diverse) overlapped a download, as already noted in the speed log.

---

## E. Validation evidence: what the tests establish

| Claim | Established by | Limits |
|---|---|---|
| Staged sampling in `run.py` equals the simulator's rule | `test_staged_matches_simulator_exactly`, on 60 random synthetic videos; real runs equal to the simulator on dev and holdout (FP32, S-3/S-4; TRT, S-10) | With PyTorch FP16 the simulation is close but not exact, because results depend on the batch (D-022). This is documented. |
| Accuracy figures | `score.py` against the every-frame FP32 reference from `build_reference.py`, which is separate code from `run.py`. The F1 maths is correct: per-class F1 is 2TP/(2TP+FP+FN); micro pools the counts; macro averages the classes that appear; the gate is min(micro, macro). | The reference uses the same decoder (PyAV) and model library, so a bug shared at the library level wouldn't be caught. The graders' reference decoder is unknown (D-014 ⚠3). |
| Dev, holdout and graded sets are disjoint | Manifest check in this audit: 0 shared videos; holdout files are outside 0–49 and outside the dev files | Files 00000 and 00001 of the graded set are in dev, by design (D-006). |
| Brief thresholds against internal targets | `score.py` `PASS` uses 0.95 / 0.85 / 0.95. D-020 targets 0.99 / 0.98 / 0.995 are checked by hand against its output. | The docs keep the two sets of thresholds apart correctly. |
| Test counts | `test_run_unit.py` has 23 `test_` functions, and all pass (re-run). The fixture manifest has 17 entries: rotation 90/180/270, VFR, nonzero start, second boundaries, timestamp gap, one frame, sub-second, empty bytes, null mp4, non-video bytes, valid after garbage, audio only, truncated moov at end, truncated faststart, corrupted middle. | The fixtures cover only the "Critical" list in D-013. High and Medium cases (other codecs, 60 fps, 1080p/4K as fixtures, multiple streams) aren't built. 4K memory behaviour is covered separately by `test_memory.py`. |
| Fixtures check the claimed behaviour | `check_fixtures.py` derives expected outputs with cv2 decoding, timestamps from the generator and single-frame predict, independently of `run.py` | Rotation correctness is checked against cv2's own auto-rotation, not a third tool. |
| Byte-identical claims | `cmp` of output files: S-5 = S-3; S-7 = S-6 (8 outputs); S-10 runs equal each other; clean environment with the old engine = S-10 (sequential set only, F-04); full-run files 00000–00001 = S-10 | Each claim holds only for the sets compared. |
| Threading is correct | `test_threaded_equals_serial_exactly` (including 1-byte queues and failure paths), `test_threaded_decoder_crash_does_not_hang`, and identical inferred-frame logs on real runs | — |
| Memory is bounded | `test_frames_released_once_second_written`, `test_buffer_bounded_for_4k_frames`, `test_oversized_second_is_streamed_not_buffered`; `test_memory.py` parts A, B and C (output only in the transcript and D-019) | The `test_memory.py` results weren't saved to files, so they can only be checked through the transcript. |
| Full run is complete | `check_full_results.py`: every row present, seconds equal 0 to the last packet's second, packet counts equal decoded frames | It uses PyAV to demux, so it isn't independent at the library level. The S-11 check ran with the old rule (seconds that have packets only); that rule would have failed on any gap video, so its pass also shows there were no gaps. The fixed rule was re-checked only on files 00000–00001, because the other 48 files were deleted. |
| Works offline | `offline_guard.py`: 0 Python network attempts | Native code isn't covered (F-05). This says nothing about whether the install works. |
| Clean install | D-027: pip resolution of every pin; a uv install from the same pins; `pip check`; `prep.py`; offline runs; fixtures; unit tests | A complete `pip install` wasn't observed (F-01). The graders' OS, driver and GPU weren't tested (F-02). |
| Behaviour on the RTX 4090 | None | It is only an expectation: the engine is built there by `prep.py`, and the fallback relies on `sm_86` code running on `sm_89`. |

---

## F. Cross-document consistency

| Topic | APPROACH.md | README.md | DECISIONS.md | HARDWARE.md | Assessment |
|---|---|---|---|---|---|
| RAM | — | — | D-001: 61 GB | 62 GiB | Inconsistent; actual 61.3 GiB (F-12). |
| Peak RSS units | 7.9 → 6.2 GB | — | 6.16, 6.23, 7.9 GB | 6.2 GB | Consistent with each other, but all in the wrong unit (F-13). |
| Clean-environment test | fresh venv, prep, runs | — | uv and cache install documented | — | APPROACH leaves out what D-027 records (F-01). |
| Old-engine byte identity | "S-10 and results.jsonl.gz" | — | "files 00000–00001" | — | APPROACH is broader than D-027 (F-04). |
| Gap seconds | — | written as empty lines (fixed at `e462731`) | D-014: written | — | Consistent with the code since `e462731`. |
| Chunk size | — | 32 seconds | 32 seconds (D-017); "CHUNK_SECONDS + 1" in the docstring | — | Off by one against the code (F-08). |
| Full-run time | 1 h 57 min 38 s | 1 h 58 min | 7,058 s | 1 h 57 min 38 s | Consistent. |
| Accuracy (shipped TRT) | 0.9928 / 0.9922 / 0.9998 dev | same | same | — | Consistent and verified against `bt3ks36vy` scores and `score.py` output. |
| Rebuilt-engine accuracy | 0.9918 dev | 0.9918 dev | 0.9918 dev; holdout 0.9935 | — | Consistent; verified against `cleantest/out/score_*.txt`. |
| Decision-log index | — | — | ends at D-025 | — | Incomplete (F-21). |

No contradictions were found in the descriptions of the sampling rule, the aggregation rule, the model settings, the fallback behaviour or the measured timings.

**Mermaid diagrams (APPROACH.md):**
1. **Sampling flow:** N appears only after stage 1, which is correct (`later_flags = flags.replace("N", "")`, `run.py:538`). The "more than one frame" condition for stage 2 is correct. The "frames remain" condition for stage 3 is missing (F-18).
2. **Optimisation timeline:** the numbers are correct.
3. **Threaded pipeline:** it shows the decoder thread running alongside the main thread, which is correct. The main thread's work (choosing frames, GPU batch, back to sampling, then the writer) is drawn as a sequential loop, which is correct. Converting frames to BGR happens in the main thread and isn't drawn, which is acceptable.
4. **Deployment:** correct. The engine path and fallback match `run.py:930-937` and `Detector.__init__`. One omission: if `prep.py` fails (for example, the engine build fails), it exits with an error; `run.py` would still run, on the PyTorch fallback, as long as `yolo11l.pt` was downloaded.

All four use quoted labels and the `<br/>` line break. The syntax is GitHub-compatible, but they were not rendered during this audit.

---

## G. Recommended corrections (replacement wording)

**G-1 (F-01)** APPROACH.md lines 293–296. Replace the "Clean-environment test" bullet with:

> - **Clean-environment test (D-027):** a fresh Python 3.11 venv and a fresh copy of the committed repository on the same server. `pip install -r requirements.txt` resolved every pinned package, but this server's route to PyPI was too slow (about 39 kB/s) for pip to finish downloading. The install was completed with `uv` from the same `requirements.txt`, using its cache and two wheels from a PyPI mirror with sha256 hashes identical to PyPI's. Then `pip check`, `prep.py` and offline runs on dev and the holdout: dev F1 0.9918, people exact 0.9920, within ±1 0.9998; holdout 0.9935 / 0.9933 / 0.9999. All 17 fixtures and 23 unit tests passed. A complete `pip install` on a fresh Ubuntu 22.04 machine has not been observed.

**G-2 (F-02)** README.md line 7:

> No `apt-get` packages are needed: PyAV ships its own FFmpeg, and OpenCV is the headless build. Tested with Python 3.11.16 on Ubuntu 26.04 with NVIDIA driver 595.99 (a Quadro RTX 6000). The pinned packages are CUDA 12.6 builds, which need a driver with CUDA 12 support; Ubuntu 22.04 and the RTX 4090 were not available for testing.

**G-3 (F-03)** README.md line 82, last sentence:

> A run on the grading machine uses its own engine, so expect its output to differ slightly from `results.jsonl.gz`. A rebuild on the same GPU model changed 0.12% of seconds; a different GPU (the RTX 4090) may differ by a different amount, which hasn't been measured.

The same idea applies to APPROACH.md line 297: "…will differ from `results.jsonl.gz`; on the same GPU model a rebuild changed 0.12% of seconds."

**G-4 (F-05)** README.md line 37:

> - The run makes no network calls of its own: `run.py` sets `YOLO_OFFLINE=1` and turns off Ultralytics' auto-install. With Python networking blocked (`validation/offline_guard.py`), the run made 0 network attempts.

**G-5 (F-06)** README.md line 54, replacement paragraph:

> If a file or video can't be read, decoding stops partway through, or the GPU runs out of memory on a single frame, the run carries on. For a video, it writes an `{"video": ..., "error": ...}` line after the lines already written. After a decode failure, every second decoded so far is still processed, including the unfinished last one (`--drop-partial-second` drops that one). After a failure during inference, seconds that were decoded but not yet written are lost. A Parquet file that can't be opened is reported as a `WARNING` on stderr; if its rows can't be read, each unread row gets an error line.

The other corrections are given in the last column of section B.

---

## H. Claims register

Each row is one substantive claim. Rows that are also findings carry the finding ID.

| # | Document | Claim | Evidence | Status |
|---|---|---|---|---|
| 1 | README | Entry point `run.py --input --output` | `run.py:900-902` | VERIFIED |
| 2 | README | YOLO11l, imgsz 640, conf 0.4, IoU 0.7, class-aware NMS | `run.py:31`; Ultralytics defaults | VERIFIED |
| 3 | README | No apt packages needed | headless cv2 (`ldd`, D-011); PyAV wheel bundles FFmpeg; tested on Ubuntu 26.04 only | PARTIALLY VERIFIED |
| 4 | README | Tested with Python 3.11 and a CUDA 12 driver | F-02 | PARTIALLY VERIFIED |
| 5 | README | `prep.py` downloads `yolo11l.pt` and checks sha256 | `prep.py:48-59`; clean-env `prep.log` | VERIFIED |
| 6 | README | `prep.py` builds a TRT FP16 engine, dynamic, batch 1–32, 640 | `prep.py` EXPORT_KW; `engine.py:366-371` | VERIFIED |
| 7 | README | `prep.py` checks a batch of 32 | `prep.py:83-91`; log | VERIFIED |
| 8 | README | Engine is tied to GPU model and TRT version, so it's built on the grader | `prep.py:68-74` metadata check; TensorRT property | VERIFIED |
| 9 | README | Build took about 6 minutes | `weights/*.json` 349.8 s; clean env 350.6 s | VERIFIED |
| 10 | README | About 9 GB of wheels | F-11 | PARTIALLY VERIFIED |
| 11 | README | TRT libs 4 GB from NVIDIA index; torch/torchvision as exact URLs with sha256 | `requirements.txt`; pip log 4,304 MB | VERIFIED |
| 12 | README | Nothing needs the network | F-05 | PARTIALLY VERIFIED |
| 13 | README | DONE summary line on stderr | `run.py:948-955` | VERIFIED |
| 14 | README | Engine missing or unloadable → warning, then PyTorch FP16 | `run.py:930-937`, `:209-214`; `s10/fallback.log` backend=pytorch | VERIFIED |
| 15 | README | Fallback results within "the accuracy figures below" | F-07 | INACCURATE |
| 16 | README | One line per second, in input order | sorted files, row order, single writer | VERIFIED |
| 17 | README | Video ID `<file>:<row>` | `run.py:715`, `:827` | VERIFIED |
| 18 | README | Second counted from the first decoded frame | `run.py:172-174` | VERIFIED |
| 19 | README | Fixed object order | `TARGETS`, `run.py:101` | VERIFIED |
| 20 | README | Gap seconds written empty | `run.py:99-100`; `test_gap_seconds_are_empty_lines`; fixture `timestamp_gap` | VERIFIED |
| 21 | README | Error handling paragraph | F-06 | PARTIALLY VERIFIED |
| 22 | README | Decoding in a separate thread, overlapped | `run.py:662-747`, `:850-894`; S-7 | VERIFIED |
| 23 | README | Queue of 512 MB | F-09 | PARTIALLY VERIFIED |
| 24 | README | Batches of up to 32 same-size frames; letterbox and NMS unchanged | `run.py:32`, `:270-275`; `predictor.py:236-242` | VERIFIED |
| 25 | README | Aggregation: at least half; floor of median | `run.py:101-102`, `:394-395` | VERIFIED |
| 26 | README | Memory: chunks of 32 seconds or 1 GB | F-08 | INACCURATE |
| 27 | README | Second over 512 MB streams through the model | `run.py:459-472`; `test_oversized_second_is_streamed_not_buffered` | VERIFIED |
| 28 | README | OOM → halve and retry | `run.py:276-289` | VERIFIED |
| 29 | README | Staged bullets: middle frame; neighbours → 3 frames; disagreement → all | `run.py:516-549` | VERIFIED |
| 30 | README | Dev 101 videos and 27,388 s; holdout 48 and 14,341 s | references; `score` output | VERIFIED |
| 31 | README | Score table: dev 0.9928 / 0.9922 / 0.9998; holdout 0.9933 / 0.9934 / 0.9999 | `bt3ks36vy` scorer output | VERIFIED |
| 32 | README | "F1 (macro gate)" label | F-10 | INACCURATE |
| 33 | README | Engine rebuild changed 0.12% of seconds; dev F1 0.9918 | `cleantest/out` status and `score_dev.txt`; 49 / 41,729 = 0.117% | VERIFIED |
| 34 | README | Same size of difference on the grading machine | F-03 | UNVERIFIABLE |
| 35 | README | Full run 1 h 58 min, about 128 h of video | `full/run.err` | VERIFIED |
| 36 | README | `results.jsonl.gz` covers files 00000–00049 | `check_full_results` (S-11) | VERIFIED |
| 37 | README | Layout: `WRITEUP.md`; `tools/` description | F-23 | OUTDATED |
| 38 | README | 17 fixtures | manifest has 17 entries | VERIFIED |
| 39 | README | `check_full_results.py` is an independent completeness check | separate code, but PyAV-based (section E) | PARTIALLY VERIFIED |
| 40 | APPROACH | Brief rules and thresholds | brief; `score.py` THRESH | VERIFIED |
| 41 | APPROACH | Reference is separate code from `run.py` | `build_reference.py`, `reference_aggregate.py` import nothing from `run.py` | VERIFIED |
| 42 | APPROACH | Test-set table: 66 / 18,806, 35 / 8,582, 48 / 14,341 | DONE lines `s10/*` | VERIFIED |
| 43 | APPROACH | Diverse files are outside the graded 50 | manifest check | VERIFIED |
| 44 | APPROACH | Holdout: nothing tuned on it | F-17 | PARTIALLY VERIFIED |
| 45 | APPROACH | Fixtures list (rotation, VFR, gaps, truncated, corrupt, empty, audio-only) | manifest names | VERIFIED |
| 46 | APPROACH | Real data has none of the unusual cases | D-012 probe (`probe_*.jsonl`) | VERIFIED |
| 47 | APPROACH | D-020 acceptance margins and the 5% rule | D-020 | VERIFIED |
| 48 | APPROACH | Baseline pipeline steps | `process_video`, `run.py:306-342` | VERIFIED |
| 49 | APPROACH | Exact `Fraction` timestamps | `run.py:170-174` | VERIFIED |
| 50 | APPROACH | Ultralytics default predict IoU is 0.5 | `utils/__init__.py:977` | VERIFIED |
| 51 | APPROACH | Batches never mix videos; batched equals single-frame (< 0.0005) | code; D-014 sample of 200 frames | VERIFIED |
| 52 | APPROACH | Baseline identical to the reference | D-015; `baseline/seq.log` | VERIFIED |
| 53 | APPROACH | 3,833 s, 137 fps | `baseline/seq.log` | VERIFIED |
| 54 | APPROACH | One CPU core, about 90% GPU | D-015 text; 105% CPU from `time -v` | PARTIALLY VERIFIED |
| 55 | APPROACH | 25.5 h estimate | F-26 | PARTIALLY VERIFIED |
| 56 | APPROACH | 67 CPU-seconds of decoding | F-19 | UNVERIFIABLE |
| 57 | APPROACH | About 50 strategies, simulated without a GPU | `sim.json` has 50 | VERIFIED |
| 58 | APPROACH | Simulator's every-frame mode reproduces the reference | `simulate_sampling.py` self-check (D-016) | VERIFIED |
| 59 | APPROACH | Strategy table values | D-016; `sim.json` | VERIFIED |
| 60 | APPROACH | Even frame counts are worse (tie rule) | values in the table; `2*count >= n` | VERIFIED |
| 61 | APPROACH | Stage-1 middle frame; N against neighbours; stage 2 three frames; V/P → all | `run.py:516-549` | VERIFIED |
| 62 | APPROACH | Stage-3 condition in the diagram | F-18 | PARTIALLY VERIFIED |
| 63 | APPROACH | Positions nest; no frame run twice | `uniform_positions`; `p not in sec.res` | VERIFIED |
| 64 | APPROACH | Aggregation uses every frame run | `run.py:551-554` | VERIFIED |
| 65 | APPROACH | Only frames sent to the model are converted | `run.py:422` | VERIFIED |
| 66 | APPROACH | `run.py` equalled the simulator on all 27,388 dev seconds (FP32) | D-017; `compare_sim` | VERIFIED |
| 67 | APPROACH | VPN FP32 scores 0.9947 / 0.9940 / 0.9999 and 0.9947 / 0.9950 / 0.9999 | D-017 and D-018 `score.py` outputs | VERIFIED |
| 68 | APPROACH | 1,049 s, 3.65×, 24.0% of frames run | `s3/seq.log`: 126,201 / 526,328 | VERIFIED |
| 69 | APPROACH | Four memory problems fixed | `run.py:459-472`, `:283-301`, `:221-227`, `:288` | VERIFIED |
| 70 | APPROACH | Profile table | `profile/p0.json` | VERIFIED |
| 71 | APPROACH | Screen, then interleave with a control | `s7`, `s8`, `s9`, `s10` logs | VERIFIED |
| 72 | APPROACH | Timeline values 3,833 / 1,049 / 658 / 534 / 310 | logs | VERIFIED |
| 73 | APPROACH | FP16 forward 1.84× faster | `p0.json` 256.0 s; `p1_fp16.json` (D-022) | VERIFIED |
| 74 | APPROACH | Tensor-core explanation | F-20 | PARTIALLY VERIFIED |
| 75 | APPROACH | FP16 object set changed on 0.070% of frames, near the threshold | D-022 `compare_precision` | VERIFIED |
| 76 | APPROACH | FP16 score table | D-022 | VERIFIED |
| 77 | APPROACH | FP16 depends on the batch (0.007); 4 of 101 videos explained | D-022 `explain_frame_diffs` | VERIFIED |
| 78 | APPROACH | FP16 658 s (−37%) | `s6` logs | VERIFIED |
| 79 | APPROACH | Threaded design: decoder thread, 512 MB queue, same main-thread code | `run.py:613-894` | VERIFIED |
| 80 | APPROACH | CPU stages 37% after FP16; PyAV releases the GIL | D-022; S-7 CPU 134% | PARTIALLY VERIFIED |
| 81 | APPROACH | Byte-identical output and frames for threaded, on every set | `s7` outputs; inferred logs | VERIFIED |
| 82 | APPROACH | Tiny-queue and crash tests | `test_threaded_equals_serial_exactly`, `test_threaded_decoder_crash_does_not_hang` | VERIFIED |
| 83 | APPROACH | Serial fallback flag | `run.py:910` | VERIFIED |
| 84 | APPROACH | 534 s (−19%) | `s7` logs | VERIFIED |
| 85 | APPROACH | Decoder threading: 4.9× decode, no gain (±0.6%), removed | `p0.json`; `s8` logs (±0.85%); no option in code | VERIFIED |
| 86 | APPROACH | TRT export settings; letterbox and NMS as PyTorch | `prep.py` EXPORT_KW; `predictor.py:236-242` | VERIFIED |
| 87 | APPROACH | TRT changed the object set on 0.123% of frames | D-025 layer 2 | VERIFIED |
| 88 | APPROACH | TRT score table | `bt3ks36vy` | VERIFIED |
| 89 | APPROACH | TRT batch-invariant | F-16 | PARTIALLY VERIFIED |
| 90 | APPROACH | Simulation exact for TRT, 0 differing seconds on both sets | `compare_sim` output | VERIFIED |
| 91 | APPROACH | 310 s (−42%), byte-identical runs | `s10` logs; `cmp` | VERIFIED |
| 92 | APPROACH | TRT 4.6 GB reserved; RAM 7.9 → 6.2 GB | F-15, F-13 | INACCURATE (units) |
| 93 | APPROACH | Batch 32 → 64 gains < 1% (FP32) | `p0.json` model_only_ips: 169.3 → 170.7 (+0.8%) | VERIFIED |
| 94 | APPROACH | #7 doesn't apply to TRT | the TRT path doesn't use cuDNN or torch for the forward pass | VERIFIED |
| 95 | APPROACH | Summary table | rows 53, 68, 78, 84, 91 | VERIFIED |
| 96 | APPROACH | Full run 1 h 57 min 38 s, 0 errors, complete | `full/run.err`; `check_full_results` | VERIFIED |
| 97 | APPROACH | 13 h estimate, about 12 min, labelled as an estimate | 715 s | VERIFIED |
| 98 | APPROACH | Engine built in prep; about 6 min after a 4 GB download | prep logs; pip log | VERIFIED |
| 99 | APPROACH | Fallback to PyTorch FP16 | rows 14 and 76 | VERIFIED |
| 100 | APPROACH | Offline: 0 attempts under the guard | `s10/offline.log`; `cleantest` logs | VERIFIED |
| 101 | APPROACH | Clean-environment test description | F-01 | PARTIALLY VERIFIED |
| 102 | APPROACH | Old engine reproduces S-10 and `results.jsonl.gz` | F-04 | PARTIALLY VERIFIED |
| 103 | APPROACH | Graders' run will differ by about this much | F-03 | INACCURATE (stated as fact; it's a prediction) |
| 104 | HARDWARE | GPU Quadro RTX 6000, 24,576 MiB; driver 595.99.02 | `nvidia-smi` | VERIFIED |
| 105 | HARDWARE | CPU i9-14900K, 24 cores, 32 threads (8P + 16E) | `lscpu`; F-24 | PARTIALLY VERIFIED |
| 106 | HARDWARE | RAM 62 GiB | F-12 | INACCURATE |
| 107 | HARDWARE | Disk WD SN580 NVMe SSD holds data and output | `lsblk`; `df` (`/dev/nvme0n1p6`) | VERIFIED |
| 108 | HARDWARE | OS Ubuntu 26.04.1, kernel 7.0.0-42; Python 3.11.16; TRT 10.16.1.11 | `lsb_release`; `uname`; venv | VERIFIED |
| 109 | HARDWARE | Full-run files, counts, times, command | `full/run.err`; `status.log` | VERIFIED |
| 110 | DECISIONS | Decision-log index complete | F-21 | OUTDATED |
| 111 | DECISIONS | Statuses and to-do items (D-011, D-014, D-020, D-027) | F-22 | OUTDATED |
| 112 | ai_chats/README | Only Claude Code used; copied unedited; no secrets found | user statement; `export_chats.py` output "secrets replaced: 0"; the snapshot is complete only up to its export time | PARTIALLY VERIFIED → re-export at the end |

**Findings not given their own register row:**
- **F-13 (memory units):** covered by rows 92 and 109, and the HARDWARE peak-memory line. Row 109 is marked VERIFIED for times and counts only.
- **F-14, F-25 and F-27:** these refine rows 109 and 78, and are counted in those rows.
- **D-025's "sized for its largest profile shape":** an inference, covered by F-15.
