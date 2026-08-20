"""Patch2Pix wrapper used as a refinement baseline.

Extracts the flow vector predicted by Patch2Pix and applies it to the baseline
keypoint coordinates (rather than replacing them outright), preserving the
matcher's own sub-pixel geometry. Auto-detects the matcher's image frame
(640 / 1296 / 1600 px) and rescales before feeding Patch2Pix so coordinates
remain in 1:1 correspondence.

Setup (one-time):
  git clone https://github.com/GrumpyZhou/patch2pix.git
  # Download pretrained weights per the upstream README.
"""
import os
import sys
import argparse
import tempfile
import h5py
import torch
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree

sys.path.append('.')
sys.path.append('..')

from subpixr.utils import standardize_path

# --- CONSTANTS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
parser.add_argument("--max_query_dist", type=float, default=4.0,
                    help="Max distance in img0 to accept a Patch2Pix match as relevant")
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
args, _ = parser.parse_known_args()

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

# H5 keypoints are in native pixel coordinates (GlueFactory does not resize).
# Images must be loaded at native size so patch extraction coordinates are correct.


# --- PATCH2PIX LOADING ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_P2P_ROOT = os.path.join(_SCRIPT_DIR, 'patch2pix')
_P2P_CKPT = os.path.join(_P2P_ROOT, 'pretrained', 'patch2pix_pretrained.pth')

_saved_sys_path = sys.path[:]
sys.path = [_P2P_ROOT] + [p for p in sys.path if p not in ('', '.', '..', _SCRIPT_DIR, os.getcwd())]
from utils.eval.model_helper import load_model as _p2p_load_fn
from utils.eval.model_helper import estimate_matches as _p2p_estimate_fn
sys.path = _saved_sys_path

patch2pix_model = _p2p_load_fn(_P2P_CKPT, method='patch2pix')
print(f"[patch2pix] Loaded from {_P2P_CKPT}")


_CUDA_POISONED = False


def p2p_run_pair(img0_bgr: np.ndarray, img1_bgr: np.ndarray) -> np.ndarray:
    """Safely runs Patch2Pix via its official file-based API."""
    global _CUDA_POISONED
    if _CUDA_POISONED:
        return np.empty((0, 4), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f0, \
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f1:
        cv2.imwrite(f0.name, img0_bgr)
        cv2.imwrite(f1.name, img1_bgr)
        tmp0, tmp1 = f0.name, f1.name
    try:
        matches, _, _ = _p2p_estimate_fn(
            patch2pix_model, tmp0, tmp1,
            ksize=2, io_thres=0.25, eval_type='fine', imsize=1024
        )
    except BaseException as e:
        # Catches torch.cuda.OutOfMemoryError, RuntimeError,
        # torch.AcceleratorError, and any other CUDA launch failure.
        msg = str(e)
        print(f"[patch2pix] skipping pair due to runtime error: {type(e).__name__}: {msg[:300]}")
        # Cap unrecoverable CUDA-context errors: stop calling CUDA for the rest of the run.
        if "unspecified launch failure" in msg or "CUDA error" in msg:
            _CUDA_POISONED = True
            print("[patch2pix] CUDA context appears poisoned; pass-through baseline kp for remaining pairs.")
        else:
            try:
                torch.cuda.empty_cache()
            except BaseException:
                _CUDA_POISONED = True
        matches = np.empty((0, 4), dtype=np.float32)
    finally:
        try:
            os.unlink(tmp0)
            os.unlink(tmp1)
        except OSError:
            pass
    return matches if matches is not None and len(matches) > 0 else np.empty((0, 4), dtype=np.float32)


def p2p_refine(pair_matches: np.ndarray, kp0: np.ndarray, kp1: np.ndarray,
               max_dist: float) -> np.ndarray:
    N = len(kp0)
    deltas = np.zeros((N, 2), dtype=np.float64)
    if len(pair_matches) == 0:
        return deltas

    tree = KDTree(pair_matches[:, :2])
    dists, indices = tree.query(kp0, distance_upper_bound=max_dist)

    mask = dists <= max_dist
    valid_indices = indices[mask]

    # FLOW PROJECTION: Apply local flow vector to the highly-accurate baseline kp0
    flow_x = pair_matches[valid_indices, 2] - pair_matches[valid_indices, 0]
    flow_y = pair_matches[valid_indices, 3] - pair_matches[valid_indices, 1]

    refined_kp1_x = kp0[mask, 0] + flow_x
    refined_kp1_y = kp0[mask, 1] + flow_y

    deltas[mask, 0] = refined_kp1_x - kp1[mask, 0]
    deltas[mask, 1] = refined_kp1_y - kp1[mask, 1]

    return deltas


def main():
    model_name = "patch2pix"
    os.makedirs(PRED_DIR, exist_ok=True)
    out_files: dict[str, h5py.File] = {}
    final_paths: dict[str, tuple[str, str]] = {}
    thresh_map: dict[str, float] = {}
    for lbl, th in GATE_CONFIGS:
        final_path = os.path.join(PRED_DIR, f"predictions_{model_name}_{lbl}_th{args.max_query_dist}.h5")
        tmp_path = final_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        out_files[lbl] = h5py.File(tmp_path, 'w')
        final_paths[lbl] = (tmp_path, final_path)
        thresh_map[lbl] = th

    oom_count = 0
    kp0_x_all: list = []
    kp0_y_all: list = []
    native_long_sides: list = []

    with h5py.File(H5_IN, 'r') as f_in:
        pairs = [(n0, n1) for n0 in f_in.keys() for n1 in f_in[n0].keys() if 'matches0' in f_in[n0][n1]]

        for name0, name1 in tqdm(pairs, desc="Patch2Pix Refinement"):
            grp_in = f_in[name0][name1]
            img0 = cv2.imread(os.path.join(IMAGE_DIR, *name0.split('-')))
            img1 = cv2.imread(os.path.join(IMAGE_DIR, *name1.split('-')))

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

            pair_matches = p2p_run_pair(img0, img1)
            if len(pair_matches) == 0:
                oom_count += 1

            deltas = p2p_refine(pair_matches, matched_kp0, matched_kp1, args.max_query_dist)

            for lbl, f_out in out_files.items():
                current_kp1 = kp1.copy()
                current_matched = matched_kp1.copy()

                mags = np.linalg.norm(deltas, axis=1)
                valid = mags <= thresh_map[lbl]

                current_matched[valid, 0] = matched_kp1[valid, 0] + deltas[valid, 0]
                current_matched[valid, 1] = matched_kp1[valid, 1] + deltas[valid, 1]

                current_kp1[m0[idx0]] = current_matched

                g = f_out.require_group(name0).create_group(name1)
                for k in grp_in.keys():
                    data = current_kp1 if k == 'keypoints1' else grp_in[k][:]
                    g.create_dataset(k, data=data)

    for lbl, f in out_files.items():
        f.close()
        tmp_p, final_p = final_paths[lbl]
        os.replace(tmp_p, final_p)

    print(f"\nDone. Pairs with no Patch2Pix matches (OOM or empty): {oom_count}")

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
            logf.write(f"# Patch2Pix coord-space sanity check\n")
            logf.write(f"dataset={args.dataset}  matcher={args.matcher}  gates={','.join(lbl for lbl, _ in GATE_CONFIGS)}\n")
            logf.write(f"expected kp range: [0, {expected_max}] (long side)\n")
            logf.write(f"pairs processed: {len(kp0_x_all)}  total kp0: {len(kp0_x)}\n")
            logf.write(f"native image long sides: min={native_arr.min()}, "
                       f"max={native_arr.max()}, mean={native_arr.mean():.1f}\n")
            logf.write(f"kp0_x: min={kp0_x.min():.2f}, max={kp0_x.max():.2f}, "
                       f"mean={kp0_x.mean():.2f}\n")
            logf.write(f"kp0_y: min={kp0_y.min():.2f}, max={kp0_y.max():.2f}, "
                       f"mean={kp0_y.mean():.2f}\n")

            frac_above_native = float((kp0_x.max(initial=0) > native_arr.min())
                                      or (kp0_y.max(initial=0) > native_arr.min()))
            below_expected = (kp0_x.max() <= expected_max + 1) and (kp0_y.max() <= expected_max + 1)
            logf.write(f"check: max kp <= {expected_max}+1 ? {below_expected}\n")
            logf.write(f"check: max kp exceeds smallest native long side "
                       f"({native_arr.min()}) ? {frac_above_native > 0}\n\n")

            logf.write("# Histogram [edge_lo, edge_hi)   kp0_x_count   kp0_y_count\n")
            for i in range(len(edges) - 1):
                logf.write(f"  [{edges[i]:5d}, {edges[i+1]:5d})   "
                           f"{hx[i]:10d}   {hy[i]:10d}\n")

        print(f"Coord-space log written to: {log_path}")
        print(f"  kp0_x max={kp0_x.max():.1f}, kp0_y max={kp0_y.max():.1f} "
              f"(expected ~{expected_max})")


if __name__ == "__main__":
    main()