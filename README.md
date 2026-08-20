# SubPixR

A generic iterative sub-pixel refiner for point tracking and feature matching.
This is the code release accompanying the BMVC 2026 submission.

SubPixR is trained once on synthetic rendered patch pairs and applied without
modification to (i) long-term point trackers (CoTracker3, TAPIR, LocoTrack) and
(ii) feature matchers (SuperPoint+LightGlue, DISK+LG, SP+SG, SP+NN, ALIKED+LG).
A closed-loop EMA scheme updates the internal template frame-by-frame for
tracker evaluations.

## Results

SubPixR is a single frozen network (9.5M params, **7.8 ms/keypoint**) applied
**without retraining** across five matchers and three trackers.

**Feature matching — HPatches, perturbed-point protocol** (MMA %, higher is
better). Among *local* refiners SubPixR is best at every threshold ≥ 1 px, while
running 2.2× faster than S2DNet and ~11× faster than COTR:

| Method | ms/kp | MMA@1 | MMA@2 | MMA@3 | MMA@5 |
|---|--:|--:|--:|--:|--:|
| Lucas–Kanade | 0.7 | 14.6 | 22.1 | 26.1 | 31.3 |
| CAPS | 13.6 | 6.8 | 22.5 | 40.4 | 60.2 |
| S2DNet | 17.5 | 17.5 | 27.1 | 30.3 | 33.5 |
| **SubPixR (ours)** | **7.8** | **27.1** | **48.4** | **58.8** | **69.9** |
| _COTR (z=3) — global re-matcher_ | _88.5_ | _31.5_ | _54.4_ | _65.7_ | _76.4_ |

SubPixR's MMA@3 of **58.8%** is **+18.4 points** above the next-best local
refiner (CAPS, 40.4%).

**Point tracking — TAP-Vid** (EPE px ↓, AJ % ↑). A closed-loop EMA template
update improves every tracker on DAVIS; for example, online mode:

| Dataset / Tracker | Base EPE | +SubPixR EPE | Base AJ | +SubPixR AJ |
|---|--:|--:|--:|--:|
| DAVIS / CoTracker3 | 7.39 | **6.64** (−10.1%) | 59.8 | **62.5** |
| DAVIS / TAPIR | 30.51 | **29.66** | 44.7 | **47.5** |
| Kinetics / LocoTrack | 16.68 | **15.39** (−7.7%) | 30.3 | **32.7** |

**Two-view pose** (Pose mAA ↑). SubPixR improves all five ScanNet matchers and
the three MegaDepth matchers with headroom — e.g. SuperPoint+LightGlue:
ScanNet 0.352 → **0.365**, MegaDepth 0.650 → **0.676**.

See the paper for the full tables and protocols.

## Repository layout

```
subpixr/            Model package: RefinementNetwork and helpers.
data/               Dataset class + synthetic-data generator (PyRender + Replica + GSO).
train.py            Main training script.
eval/               Evaluation scripts:
  tapvid_closed_loop.py  Unified closed-loop TAP-Vid eval (--tracker cotracker|tapir|locotrack).
  hpatches_*.py      HPatches (homography baselines + perturbed-point Table 4).
  pose_megadepth_scannet.py  Two-view pose mAA across five matchers.
  benchmark_speed.py Figure 5 latency curves.
refiners_compared/  Wrappers for prior-art refiners (S2DNet, Patch2Pix, COTR).
scripts/            Bash runners for the four evaluation tracks + training.
```

## Setup

### 1. Install dependencies

PyTorch first (match your CUDA version):
```
pip install torch torchvision  # or follow https://pytorch.org/get-started/locally/
```

Then everything else:
```
pip install -r requirements.txt
```

### 2. SubPixR checkpoint (included)

The released weights ship with this repo at **`checkpoints/best_model.pth`**
(best architecture from the paper — ResNet-34 + cross-correlation + attention +
hybrid fusion; the exact training config is in `checkpoints/config.json`). The
eval and training scripts load it automatically — nothing to download.

To use a different checkpoint, point `SUBPIXR_CHECKPOINT` at it (or pass
`--refiner_path` to any script):
```
export SUBPIXR_CHECKPOINT=/path/to/your_model.pth
```

### 3. Clone the third-party SOTA repos you need

Each evaluation track depends on the upstream code of the baseline matchers and
trackers. Clone the ones you intend to run.

**For TAP-Vid evaluation (one or more of):**
| Tracker | Repo | Notes |
|---|---|---|
| CoTracker3 | https://github.com/facebookresearch/co-tracker | Clone into `eval/co-tracker/`. Checkpoint: `scaled_offline.pth` from the [release page](https://github.com/facebookresearch/co-tracker/releases). |
| TAPIR | https://github.com/google-deepmind/tapnet | Pip-install per upstream README. Checkpoint: `bootstapir_checkpoint_v2.pt`. |
| LocoTrack | https://github.com/cvlab-kaist/locotrack | Clone next to the eval scripts. Checkpoint: `locotrack_small.pt`. |

**For two-view pose evaluation:**
| Component | Repo | Notes |
|---|---|---|
| glue-factory (driver) | https://github.com/cvg/glue-factory | Run the per-matcher baseline export to produce `predictions_baseline.h5`. |
| LightGlue | https://github.com/cvg/LightGlue | Pulled by glue-factory. |
| SuperGlue | https://github.com/magicleap/SuperGluePretrainedNetwork | Weights need a separate license — see upstream. |
| DISK | https://github.com/cvlab-epfl/disk | Pulled by glue-factory. |
| ALIKED | https://github.com/Shiaoming/ALIKED | Pulled by glue-factory. |

**For the prior-art refiner comparisons (optional — only if you want to reproduce Tables 4 + 5 rows for them):**
| Refiner | Repo | Weights |
|---|---|---|
| S2DNet | https://github.com/germain-hug/S2DNet-Minimal | Per upstream README. |
| Patch2Pix | https://github.com/GrumpyZhou/patch2pix | Per upstream `download.sh`. |
| COTR | https://github.com/ubc-vision/COTR | `cotr_default.zip` from the GitHub releases. |
| CAPS | https://github.com/qianqianwang68/caps | Per upstream README. |
| LoFTR | https://github.com/zju3dv/LoFTR | Per upstream README. |
| ASpanFormer | https://github.com/apple/ml-aspanformer | Per upstream README. |

**For HPatches evaluation:**
| Dataset | Source |
|---|---|
| HPatches sequences | https://github.com/hpatches/hpatches-dataset (full sequences, 116 scenes). |

### 4. Point at your datasets

Tell SubPixR where the TAP-Vid `.pkl` files live (only needed if you run TAP-Vid eval):
```
export TAPVID_ROOT=/path/to/image_databases
# expected layout:
#   $TAPVID_ROOT/tapvid_davis/tapvid_davis.pkl
#   $TAPVID_ROOT/tapvid_kinetics/test/          (folder of .pkl shards)
#   $TAPVID_ROOT/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl
```

Get the TAP-Vid `.pkl` files from https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid.

## Running the evaluations

All commands assume you are inside `release/subpixr/`.

### TAP-Vid (Table 3 in the paper)

All three trackers (CoTracker3, LocoTrack, TAPIR) run through one unified
closed-loop script, `eval/tapvid_closed_loop.py`, selected with `--tracker`:
```
# all three trackers, closed-loop, online (causal):
bash scripts/eval_tapvid_all.sh davis            # or kinetics / rgb_stacking
bash scripts/eval_tapvid_all.sh davis offline    # full-video (offline) mode

# a single tracker directly:
PYTHONPATH=. python3 eval/tapvid_closed_loop.py davis \
    --tracker cotracker --mode online --ema_alpha 0.9
```
Pass a checkpoint with `--refiner_path /path/to/best_model.pth` (or as the 4th
positional arg to the runner) to override `$SUBPIXR_CHECKPOINT`. The EMA momentum
`--ema_alpha 0.9` is the paper default.

### HPatches (Table 4)
```
bash scripts/eval_hpatches.sh
```

### Two-view pose, MegaDepth + ScanNet (Table 5)
```
bash scripts/eval_pose_all.sh                 # SubPixR only
bash scripts/eval_pose_all.sh --sota          # + S2DNet / Patch2Pix / COTR
```
This expects the baseline `predictions_baseline.h5` files to already exist (run
glue-factory's per-matcher export first; see its README).

### Inference-time benchmark (Figure 5)
```
PYTHONPATH=. python3 eval/benchmark_speed.py
```

## Training from scratch

```
bash scripts/train.sh
```

This trains the final architecture (ResNet-34, depthwise cross-correlation,
cross-attention bottleneck, hybrid fusion, `train_iters=4`) on the synthetic
patch dataset. Edit the `EXPERIMENTS` array to queue ablations.

### Generating the synthetic patch dataset

You need:
- **Google Scanned Objects** (3D meshes): https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research
- **Replica** scenes (Habitat-format): https://github.com/facebookresearch/Replica-Dataset

Edit the path fields in the `Config` dataclass at the top of
`data/generate_synthetic_dataset.py`, then:
```
PYTHONPATH=. python3 data/generate_synthetic_dataset.py
```
Output goes to `OUT_ROOT` (default `~/subpixr_synth/`).
PyRender needs the EGL backend on headless Linux; the script sets
`PYOPENGL_PLATFORM=egl` automatically.

### Logging to Weights & Biases (optional)
```
export WANDB_API_KEY=<your_key>
```
W&B logging is disabled on Windows and when the env var is unset.

## Citation

If you use SubPixR, please cite the BMVC paper:
```
@inproceedings{ileni2026subpixr,
  title     = {SubPixR: A Generic Iterative Sub-Pixel Refiner for Point Tracking and Feature Matching},
  author    = {Ileni, Tudor Alexandru and Darabant, Adrian Sergiu and Maduta, Adrian Pavel},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

## License

Copyright 2026 The SubPixR Authors.

The code and the trained model are released under **different terms**, because
they carry different obligations. Full text in `LICENSE`.

| What | Terms | Commercial use |
|---|---|---|
| Source code | [MIT](https://opensource.org/license/mit) | **Yes** |
| Model weights (`checkpoints/`) | [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) | **No** |
| `eval/tapvid_metrics.py`, `eval/transforms.py` | Apache 2.0, © Google LLC | Yes |

The weights are the restricted part: SubPixR is trained on images rendered from
the [Replica dataset](https://github.com/facebookresearch/Replica-Dataset) and
[Google Scanned Objects](https://app.gazebosim.org/GoogleResearch). Replica's
Research Terms permit non-commercial research and educational use only, and
that restriction carries over to models trained on renders of the data — so we
cannot grant commercial rights in the checkpoint, and nobody downstream can
either. The code contains none of those assets, so it is free of that
constraint: use it commercially, retrain it on your own data, ship it.

Third-party trackers, matchers, and refiners cloned for evaluation are governed
by their own licenses; Part D of `LICENSE` lists them along with the full data
provenance.
