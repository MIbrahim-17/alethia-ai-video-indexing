# Hardware

The full 50 files were processed on a **university machine**, a Linux workstation reached over SSH. The run used only one of its two GPUs, GPU 0.

| | |
|---|---|
| GPU | NVIDIA Quadro RTX 6000 (Turing), 24 GB (24,576 MiB). The machine has two; the run used one. |
| CPU | Intel Core i9-14900K: 24 cores, 32 threads (`lscpu`); 8 performance and 16 efficiency cores, per Intel's specification |
| RAM | 61.3 GiB usable (65.8 GB), as reported by `free -b` |
| Disk | SSD: WD Blue SN580 1 TB NVMe. The data and the output were on this disk. |
| OS | Ubuntu 26.04.1 LTS, Linux 7.0.0-42 |
| Driver / CUDA | NVIDIA driver 595.99.02. PyTorch 2.14.1 with CUDA 12.6, and TensorRT 10.16.1.11. |
| Python | 3.11.16 |

## Full run

- **Files:** `train-00000-of-01357.parquet` to `train-00049-of-01357.parquet`: 1,650 videos, 461,757 seconds, about 128 h of video.
- **Command:** `python run.py --input data/finevideo/data --output results.jsonl`, with the TensorRT engine from `python prep.py`.
- **Time from start to finish: 1 h 57 min 38 s**, from 08:00:32 to 09:58:11 PKT on 2026-11-21, measured with `/usr/bin/time` around the whole process.
  - `run.py`'s own timer reported 7,058 s.
- **Peak memory:** 5.9 GiB of RAM (resident; 6,159,932 KiB from `/usr/bin/time`). GPU memory was about 5.7 GiB (5,843 MiB in `nvidia-smi` snapshots every 10 minutes; not a continuous peak measurement).
- **Other load:** no other job was started on GPU 0 during the run (checked in the `nvidia-smi` snapshots every 10 minutes).
