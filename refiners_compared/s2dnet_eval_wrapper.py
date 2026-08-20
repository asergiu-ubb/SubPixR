"""S2DNet sparse-to-dense refinement baseline.

Wraps the published S2DNet pipeline and runs it over a matcher's
predictions_baseline.h5 file. Auto-detects the matcher's image frame
(640 / 1296 / 1600 px) and resizes the image to match before extracting
hyperfeatures, so S2DNet's feature grid stays aligned with the matcher's
sub-pixel keypoints.

Setup (one-time):
  git clone https://github.com/germain-hug/S2DNet-Minimal.git
  # Download the released weights per the upstream README.
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
sys.path.append('..')

from subpixr.utils import standardize_path
from s2dnet_refiner import load_s2dnet, extract_pyramid, refine_hierarchical

# --- CONSTANTS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEARCH_RADIUS = 64  # ±64px window around coarse estimate (128px total, same as refiner)

# --- ARGUMENTS ---
_MATCHERS = [
    "superpoint+lightglue-official",
    "superpoint+superglue-official",
    "superpoint+NN",
    "disk+lightglue-official",
    "aliked+lightglue-official",
]
parser = argparse.ArgumentParser()
parser.add_argument("--skip_debug", action="store_true", default=True)
parser.add_argument("--search_radius", type=int, default=64,
                    help="Half-window size for cosine similarity search (px)")
parser.add_argument("--dataset", type=str, default="megadepth",
                    choices=["megadepth", "scannet"],
                    help="Evaluation dataset.")
parser.add_argument("--matcher", type=str, default="superpoint+lightglue-official",
                    choices=_MATCHERS,
                    help="Matcher whose predictions_baseline.h5 to refine.")
parser.add_argument("--gate_label", type=str, default=None,
                    choices=["gate_5px", "gate_10px", "no_gate"],
                    help="Single gate label (deprecated — prefer --gate_labels).")
parser.add_argument("--gate_labels", type=str, nargs='+', default=None,
                    choices=["gate_5px", "gate_10px", "no_gate"],
                    help="One or more gate labels to produce in a single model pass.")
parser.add_argument("--cycle_thresh", type=float, default=4.0,
                    help="Max round-trip pixel error for cycle-consistency filter.")
parser.add_argument("--mid_radius_px", type=float, default=32.0,
                    help="Image-pixel radius for the conv3_3 (mid) search window.")
parser.add_argument("--fine_radius_px", type=float, default=4.0,
                    help="Image-pixel radius for the conv1_2 (fine) search window.")
args, _ = parser.parse_known_args()
SEARCH_RADIUS = args.search_radius

_GATE_TO_THRESH = {"gate_5px": 5.0, "gate_10px": 10.0, "no_gate": 100.0}
_gate_labels: list[str] = list(args.gate_labels) if args.gate_labels else []
if args.gate_label and args.gate_label not in _gate_labels:
    _gate_labels.append(args.gate_label)
if not _gate_labels:
    _gate_labels = ["gate_5px"]
GATE_CONFIGS: list[tuple[str, float]] = [(lbl, _GATE_TO_THRESH[lbl]) for lbl in _gate_labels]

# --- CONFIGURATION ---
ROOT_DIR  = standardize_path("./glue-factory")
_DS_FOLDER = f"{args.dataset}1500"
PRED_DIR  = os.path.join(ROOT_DIR, "outputs/results", _DS_FOLDER, args.matcher)
H5_IN     = os.path.join(PRED_DIR, "predictions_baseline.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "data", _DS_FOLDER,
                         "images" if args.dataset == "megadepth" else "")

# --- S2DNET LOADING ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_S2D_ROOT = os.path.join(_SCRIPT_DIR, 'S2DNet-Minimal')
_S2D_CKPT = os.path.join(_S2D_ROOT, 's2dnet_weights.pth')

s2dnet_model, _LEVEL_SCALES = load_s2dnet(_S2D_CKPT, DEVICE)
print(f"[S2DNet] Loaded from {_S2D_CKPT}")
print(f"[S2DNet] Pyramid layers from checkpoint: {list(s2dnet_model._hypercolumn_layers)}")
print(f"[S2DNet] Level strides: {_LEVEL_SCALES}")


# H5 keypoints are in native pixel coordinates (GlueFactory does not resize).
# Images must be loaded at native size so patch extraction coordinates are correct.


def s2d_extract_pyramid(img_bgr: np.ndarray) -> list:
    """Extract S2DNet's full hyperfeature pyramid from a BGR image."""
    img_rgb = img_bgr[:, :, ::-1].copy()
    pyr = extract_pyramid(img_rgb, s2dnet_model, DEVICE)
    return [f.cpu() for f in pyr]


def main():
    model_name = "s2dnet"
    os.makedirs(PRED_DIR, exist_ok=True)
    out_files: dict[str, h5py.File] = {}
    final_paths: dict[str, tuple[str, str]] = {}
    thresh_map: dict[str, float] = {}
    for lbl, th in GATE_CONFIGS:
        final_path = os.path.join(PRED_DIR, f"predictions_{model_name}_{lbl}.h5")
        tmp_path = final_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        out_files[lbl] = h5py.File(tmp_path, 'w')
        final_paths[lbl] = (tmp_path, final_path)
        thresh_map[lbl] = th

    kp0_x_all: list = []
    kp0_y_all: list = []
    native_long_sides: list = []

    with h5py.File(H5_IN, 'r') as f_in:
        pairs = [(n0, n1) for n0 in f_in.keys() for n1 in f_in[n0].keys() if 'matches0' in f_in[n0][n1]]

        feat_cache: dict[str, list] = {}

        for name0, name1 in tqdm(pairs, desc="S2DNet Hierarchical Refinement"):
            grp_in = f_in[name0][name1]
            path0 = os.path.join(IMAGE_DIR, *name0.split('-'))
            path1 = os.path.join(IMAGE_DIR, *name1.split('-'))
            img0 = cv2.imread(path0)
            img1 = cv2.imread(path1)

            if img0 is None or img1 is None:
                for f_out in out_files.values():
                    g = f_out.require_group(name0).create_group(name1)
                    for k in grp_in.keys():
                        g.create_dataset(k, data=grp_in[k][:])
                continue

            native_long_sides.append(max(img0.shape[:2]))
            native_long_sides.append(max(img1.shape[:2]))

            # No resize: images stay at native size, matching the H5 keypoint frame.

            kp0, kp1, m0 = grp_in['keypoints0'][:], grp_in['keypoints1'][:], grp_in['matches0'][:]
            kp0_x_all.append(kp0[:, 0])
            kp0_y_all.append(kp0[:, 1])
            idx0 = np.where(m0 > -1)[0]
            matched_kp0, matched_kp1 = kp0[idx0], kp1[m0[idx0]]

            if len(matched_kp0) == 0:
                for f_out in out_files.values():
                    g = f_out.require_group(name0).create_group(name1)
                    for k in grp_in.keys():
                        g.create_dataset(k, data=grp_in[k][:])
                continue

            if name0 not in feat_cache:
                feat_cache[name0] = s2d_extract_pyramid(img0)
            if name1 not in feat_cache:
                feat_cache[name1] = s2d_extract_pyramid(img1)

            pyr0 = [f.to(DEVICE) for f in feat_cache[name0]]
            pyr1 = [f.to(DEVICE) for f in feat_cache[name1]]

            deltas, cycle_ok = refine_hierarchical(
                pyr0, pyr1, matched_kp0, matched_kp1, _LEVEL_SCALES,
                mid_radius_px=args.mid_radius_px,
                fine_radius_px=args.fine_radius_px,
                cycle_thresh=args.cycle_thresh,
            )

            for lbl, f_out in out_files.items():
                current_kp1 = kp1.copy()
                current_matched = matched_kp1.copy()

                mags = np.linalg.norm(deltas, axis=1)
                valid = (mags <= thresh_map[lbl]) & cycle_ok

                current_matched[valid, 0] = matched_kp1[valid, 0] + deltas[valid, 0]
                current_matched[valid, 1] = matched_kp1[valid, 1] + deltas[valid, 1]

                current_kp1[m0[idx0]] = current_matched

                g = f_out.require_group(name0).create_group(name1)
                for k in grp_in.keys():
                    data = current_kp1 if k == 'keypoints1' else grp_in[k][:]
                    g.create_dataset(k, data=data)

            if len(feat_cache) > 4:
                feat_cache.clear()
                torch.cuda.empty_cache()

    for lbl, f in out_files.items():
        f.close()
        tmp_p, final_p = final_paths[lbl]
        os.replace(tmp_p, final_p)

    # --- Coordinate-space sanity histogram on baseline kp0 ---
    if kp0_x_all:
        kp0_x = np.concatenate(kp0_x_all)
        kp0_y = np.concatenate(kp0_y_all)
        native_arr = np.asarray(native_long_sides)

        expected_max = int(max(kp0_x.max(), kp0_y.max())) + 1
        bin_step = 100 if expected_max > 1000 else 50
        edges = np.arange(0, expected_max + (bin_step * 2), bin_step)

        hx, _ = np.histogram(kp0_x, bins=edges)
        hy, _ = np.histogram(kp0_y, bins=edges)

        log_path = list(final_paths.values())[0][1].replace(".h5", ".log")
        with open(log_path, "w") as logf:
            logf.write(f"# S2DNet coord-space sanity check\n")
            logf.write(f"dataset={args.dataset}  matcher={args.matcher}  gates={','.join(lbl for lbl, _ in GATE_CONFIGS)}\n")
            logf.write(f"expected kp range: [0, {expected_max}] (long side)\n")
            logf.write(f"pairs processed: {len(kp0_x_all)}  total kp0: {len(kp0_x)}\n")
            logf.write(f"native image long sides: min={native_arr.min()}, "
                       f"max={native_arr.max()}, mean={native_arr.mean():.1f}\n")
            logf.write(f"kp0_x: min={kp0_x.min():.2f}, max={kp0_x.max():.2f}, "
                       f"mean={kp0_x.mean():.2f}\n")
            logf.write(f"kp0_y: min={kp0_y.min():.2f}, max={kp0_y.max():.2f}, "
                       f"mean={kp0_y.mean():.2f}\n")

            below_expected = (kp0_x.max() <= expected_max + 1) and (kp0_y.max() <= expected_max + 1)
            exceeds_min_native = (kp0_x.max() > native_arr.min()) or (kp0_y.max() > native_arr.min())
            logf.write(f"check: max kp <= {expected_max}+1 ? {below_expected}\n")
            logf.write(f"check: max kp exceeds smallest native long side "
                       f"({native_arr.min()}) ? {exceeds_min_native}\n\n")

            logf.write("# Histogram [edge_lo, edge_hi)   kp0_x_count   kp0_y_count\n")
            for i in range(len(edges) - 1):
                logf.write(f"  [{edges[i]:5d}, {edges[i+1]:5d})   "
                           f"{hx[i]:10d}   {hy[i]:10d}\n")

        print(f"Coord-space log written to: {log_path}")
        print(f"  kp0_x max={kp0_x.max():.1f}, kp0_y max={kp0_y.max():.1f} "
              f"(expected ~{expected_max})")

    print("\nDone.")


if __name__ == "__main__":
    main()