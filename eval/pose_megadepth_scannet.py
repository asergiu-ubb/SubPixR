"""Two-view pose evaluation on MegaDepth-1500 and ScanNet-1500.

Drives an arbitrary baseline matcher (e.g. SuperPoint+LightGlue, DISK+LG, SP+SG,
ALIKED+LG, SP+NN), optionally refines the matches with SubPixR or one of the
prior-art refiners, then computes pose mAA via the official protocols.

Notes:
- ScanNet matching is run at its native 1296px resolution.
- Patches are resized first, then cropped; image and keypoint frames stay 1:1.
- The INNER_GATE setting protects SfM pseudo-GT from sub-pixel jitter.
"""

import os
import re
import sys
import shutil
import argparse
import subprocess
import h5py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image as _PILImage

sys.path.append('.')
sys.path.append('..')

from subpixr.utils import standardize_path, get_general_refiner, load_refiner

# --- CONSTANTS ---
_NORM_MEAN_T = torch.tensor([0.485, 0.456, 0.406])
_NORM_STD_T  = torch.tensor([0.229, 0.224, 0.225])
THRESHOLDS   = [5.0]  # overwritten when --gate_sweep is passed
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="megadepth", choices=["megadepth", "scannet"])
parser.add_argument("--matcher", type=str, default="superpoint+lightglue-official")
parser.add_argument("--refiner_path", type=str, default=None, help="Path to refiner checkpoint")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_iters", type=int, default=1)
parser.add_argument("--gate_label", type=str, default="gate_5px")
parser.add_argument("--inner_gate", type=float, default=0,
                    help="Deadzone filter: ignore refinements smaller than this (px). Default 0.5.")
parser.add_argument("--gate_sweep", action="store_true",
                    help="Write one predictions H5 per gate in {5,10,20,inf}px.")
parser.add_argument("--gates", type=str, default=None,
                    help="Comma-separated list of gate thresholds in px (e.g. '5,10').")
parser.add_argument("--frame_diag_max", type=int, default=0,
                    help="Print img.shape vs kpts max/min for the first N pairs to "
                         "diagnose H5-keypoint vs cv2-resize frame mismatches "
                         "(uniform-shift symptom).")
parser.add_argument("--skip_eval", action="store_true",
                    help="Skip the standalone GlueFactory eval + H5 cleanup (results_*.txt).")

args, _ = parser.parse_known_args()

if args.gates is not None:
    THRESHOLDS = [float("inf") if s.strip().lower() == "inf" else float(s)
                  for s in args.gates.split(",") if s.strip()]
elif args.gate_sweep:
    THRESHOLDS = [5.0, 10.0, 20.0, float("inf")]


def _gate_label(th: float, inner: float) -> str:
    # Format inner_gate to drop trailing zero if integer (0.5 -> 0.5, 1.0 -> 1)
    ig_str = f"{inner:.1f}".rstrip('0').rstrip('.')
    if not np.isfinite(th):
        return f"gate_inf_ig{ig_str}"
    return f"gate_{int(round(th))}px_ig{ig_str}"

# --- CONFIGURATION ---
ROOT_DIR   = standardize_path("./glue-factory")
_DS_FOLDER = f"{args.dataset}1500"
PRED_DIR   = os.path.join(ROOT_DIR, "outputs/results", _DS_FOLDER, args.matcher)
H5_IN      = os.path.join(PRED_DIR, "predictions_baseline.h5")
IMAGE_DIR  = os.path.join(ROOT_DIR, "data", _DS_FOLDER, "images" if args.dataset == "megadepth" else "")

# --- HELPERS ---
def key_to_image_path(key: str) -> str:
    return os.path.join(IMAGE_DIR, *key.split('-'))


# def auto_detect_kp_frame(h5_path: str) -> int:
#     max_val = 0.0
#     with h5py.File(h5_path, 'r') as f:
#         for n0 in f.keys():
#             for n1 in f[n0].keys():
#                 if 'keypoints0' in f[n0][n1]:
#                     kp0 = f[n0][n1]['keypoints0'][:]
#                     if len(kp0) > 0:
#                         max_val = max(max_val, float(kp0.max()))
#
#     if max_val <= 640 + 50:
#         return 640
#     elif max_val <= 1024 + 50:
#         return 1024
#     elif max_val <= 1296 + 50:
#         return 1296
#     else:
#         return 1600


# TARGET_LONG = auto_detect_kp_frame(H5_IN)
# print(f"Auto-detected Baseline Keypoint Frame: {TARGET_LONG}px")


# def gf_target_size(h: int, w: int, side_size: int = None,
#                    side: str = "long", edge_divisible_by: int | None = None
#                    ) -> tuple[int, int]:
#     """Replicate GlueFactory's ImagePreprocessor.get_new_image_size EXACTLY.
#
#     Critical: GF uses int() (truncation), NOT int(round()). Mismatched
#     rounding here vs in GF's H5 export causes a sub-pixel-to-multi-pixel
#     shift between the loaded image and the stored keypoints.
#     Returns (H, W).
#     """
#     if side_size is None:
#         side_size = TARGET_LONG
#     aspect_ratio = w / h
#     if side == "long":
#         if aspect_ratio < 1.0:                 # portrait
#             size_hw = (side_size, int(side_size * aspect_ratio))
#         else:                                  # landscape
#             size_hw = (int(side_size / aspect_ratio), side_size)
#     elif side == "short":
#         if aspect_ratio < 1.0:
#             size_hw = (int(side_size / aspect_ratio), side_size)
#         else:
#             size_hw = (side_size, int(side_size * aspect_ratio))
#     else:
#         raise ValueError(f"Unsupported side={side}")
#     if edge_divisible_by is not None:
#         df = edge_divisible_by
#         size_hw = (int(size_hw[0] // df * df), int(size_hw[1] // df * df))
#     return size_hw


def load_and_resize_to_baseline(path: str) -> np.ndarray:
    """Load image at NATIVE size — no resize.

    GlueFactory's eval configs for megadepth1500 / scannet1500 set only
    `preprocessing.side='long'` and leave `preprocessing.resize=None`,
    so ImagePreprocessor SKIPS the resize entirely (see image.py:37).
    H5 keypoints are therefore in each image's native pixel frame, NOT
    in a 1600px-long-side frame.

    Previously this function upscaled small images to TARGET_LONG which
    placed already-correct kpts off their image content (uniform-shift
    bug visible in samples_claude/0057 — kpts floated above the dome
    because img1 native was ~1268x947 but we resized to 1600x1199).
    """
    return cv2.imread(path)


def batch_crop_gpu(frame_bgr: np.ndarray, kpts: np.ndarray, size: int) -> torch.Tensor:
    H, W = frame_bgr.shape[:2]
    N = len(kpts)

    img_t = torch.from_numpy(frame_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().to(DEVICE) / 255.0
    img_t = ((img_t - _NORM_MEAN_T.to(DEVICE)[:, None, None]) / _NORM_STD_T.to(DEVICE)[:, None, None])

    xs = torch.from_numpy(kpts[:, 0]).to(DEVICE).float()
    ys = torch.from_numpy(kpts[:, 1]).to(DEVICE).float()

    grid_y, grid_x = torch.meshgrid(torch.arange(size, device=DEVICE), torch.arange(size, device=DEVICE), indexing='ij')
    grid_y = grid_y.float() - size / 2.0 + 0.5
    grid_x = grid_x.float() - size / 2.0 + 0.5

    sample_x = xs.view(-1, 1, 1) + grid_x.view(1, size, size)
    sample_y = ys.view(-1, 1, 1) + grid_y.view(1, size, size)

    sample_x = (sample_x / (W - 1)) * 2.0 - 1.0
    sample_y = (sample_y / (H - 1)) * 2.0 - 1.0

    grid = torch.stack((sample_x, sample_y), dim=3)
    img_batch = img_t.unsqueeze(0).expand(N, -1, -1, -1)

    return F.grid_sample(img_batch, grid, align_corners=True, padding_mode='zeros')


def _cpu_crop(img_bgr: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    half = size // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + size, y0 + size
    out = np.zeros((size, size, 3), dtype=img_bgr.dtype)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x1), min(H, y1)
    w, h = dx1 - dx0, dy1 - dy0
    if w > 0 and h > 0:
        out[sy0:sy0 + h, sx0:sx0 + w] = img_bgr[dy0:dy1, dx0:dx1]
    return out


def run_gf_eval_and_log(dataset: str, matcher: str, model_name: str,
                        label: str, ckpt_dir: str) -> bool:
    results_dir = os.path.join(ROOT_DIR, "outputs/results", f"{dataset}1500")
    pred_dir = os.path.join(results_dir, matcher)
    src_h5 = os.path.join(pred_dir, f"predictions_{model_name}_{label}.h5")
    if not os.path.exists(src_h5):
        print(f"  [!] Missing H5 for GF eval: {src_h5}")
        return False

    exp_name = f"eval_{model_name}_{matcher}_{label}"
    exp_dir = os.path.join(results_dir, exp_name)
    target_h5 = os.path.join(exp_dir, "predictions.h5")
    result_log = os.path.join(ckpt_dir, f"results_{dataset}_{matcher}_{label}.txt")
    gf_module = f"gluefactory.eval.{dataset}1500"

    shutil.rmtree(exp_dir, ignore_errors=True)
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy(src_h5, target_h5)

    print(f"  -> GF eval {dataset}/{matcher} ({label}) ...")
    with open(result_log, "w", encoding="utf-8") as fout:
        proc = subprocess.run(
            [sys.executable, "-m", gf_module,
             "--conf", "superpoint+lightglue-official",
             "--tag", exp_name],
            cwd=ROOT_DIR,
            stdout=fout, stderr=subprocess.STDOUT,
        )

    try:
        with open(result_log, "r", encoding="utf-8", errors="ignore") as fr:
            content = fr.read()
        with open(result_log, "w", encoding="utf-8") as fw:
            fw.write(_ANSI_RE.sub("", content))
    except OSError:
        pass

    shutil.rmtree(exp_dir, ignore_errors=True)

    ok = proc.returncode == 0 and "rel_pose_error_mAA" in open(
        result_log, "r", encoding="utf-8", errors="ignore").read()
    status = "ok" if ok else f"FAILED (exit {proc.returncode})"
    print(f"     log: {result_log}  [{status}]")
    return ok


def load_scannet_calib(calib_txt: str) -> dict:
    """Load ScanNet pairs_calibrated.txt.

    Format per line:  name0 name1  K0(9)  K1(9)  T(16)
    where T is a 4×4 row-major [R|t; 0 1] matrix.
    Keys are stored under both the native '/' form and the '-' form used
    by GlueFactory H5 files so either lookup style works.
    """
    calib: dict = {}
    with open(calib_txt, 'r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 36:          # 2 names + 9 + 9 + 16
                continue
            n0, n1 = parts[0], parts[1]
            vals = np.array(parts[2:], dtype=np.float64)
            K0 = vals[0:9].reshape(3, 3)
            K1 = vals[9:18].reshape(3, 3)
            T  = vals[18:34].reshape(4, 4)
            R, t = T[:3, :3], T[:3, 3]
            entry = {'K0': K0, 'K1': K1, 'R': R, 't': t}
            calib[(n0, n1)] = entry
            n0h, n1h = n0.replace('/', '-'), n1.replace('/', '-')
            if (n0h, n1h) != (n0, n1):
                calib[(n0h, n1h)] = entry
    return calib


def load_megadepth_calib(calib_txt: str) -> dict:
    calib: dict = {}
    with open(calib_txt, 'r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 32:
                continue
            n0, n1 = parts[0], parts[1]
            vals = np.array(parts[2:], dtype=np.float64)
            K0 = vals[0:9].reshape(3, 3)
            K1 = vals[9:18].reshape(3, 3)
            R  = vals[18:27].reshape(3, 3)
            t  = vals[27:30]
            entry = {'K0': K0, 'K1': K1, 'R': R, 't': t}
            calib[(n0, n1)] = entry
            # H5 keys use '-' as path separator; store both forms so lookups work
            n0h, n1h = n0.replace('/', '-'), n1.replace('/', '-')
            if (n0h, n1h) != (n0, n1):
                calib[(n0h, n1h)] = entry
    return calib


def get_image_original_size(path: str):
    try:
        with _PILImage.open(path) as im:
            w, h = im.size
        return h, w
    except Exception:
        return None


def _scale_K(K: np.ndarray, orig_H: int, orig_W: int) -> np.ndarray:
    scale = TARGET_LONG / float(max(orig_H, orig_W))
    Ks = K.copy()
    Ks[0] *= scale
    Ks[1] *= scale
    return Ks


def _compute_F_from_gt(K0: np.ndarray, K1: np.ndarray,
                        R: np.ndarray, t: np.ndarray) -> np.ndarray:
    tx, ty, tz = t
    t_cross = np.array([[ 0,  -tz,  ty],
                         [ tz,   0, -tx],
                         [-ty,  tx,   0]], dtype=np.float64)
    E = t_cross @ R
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    return F


def _epi_dist(F: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    N = len(p0)
    p0h = np.hstack([p0, np.ones((N, 1), dtype=np.float64)])
    p1h = np.hstack([p1, np.ones((N, 1), dtype=np.float64)])
    l1  = (F @ p0h.T).T
    num = np.abs(np.einsum('ij,ij->i', p1h, l1))
    den = np.sqrt(l1[:, 0]**2 + l1[:, 1]**2 + 1e-12)
    return num / den


def _passthrough(out_files, name0, name1, grp_in):
    for f_out in out_files.values():
        g = f_out.require_group(name0).create_group(name1)
        for k in grp_in.keys():
            g.create_dataset(k, data=grp_in[k][:])


def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    # Load Refiner
    if args.refiner_path is None:
        refiner, scale_factor = load_refiner(get_general_refiner(), DEVICE)
    else:
        refiner, scale_factor = load_refiner(args.refiner_path, DEVICE)
    refiner.eval()

    # Output Files Setup
    out_files = {}
    final_paths = {}
    model_name = os.path.basename(os.path.dirname( (os.path.abspath(args.refiner_path) if (args.refiner_path is not None) else get_general_refiner())))
    for th in THRESHOLDS:
        # Changed this logic so args.inner_gate is used for args.gate_label too!
        label = _gate_label(th,args.inner_gate) if len(THRESHOLDS) > 1 else f"{args.gate_label}_ig{args.inner_gate:.1f}".rstrip('0').rstrip('.')
        final_path = os.path.join(PRED_DIR, f"predictions_{model_name}_{label}.h5")
        tmp_path = final_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        out_files[th] = h5py.File(tmp_path, 'w')
        final_paths[th] = (tmp_path, final_path)

    ckpt_path = os.path.abspath(args.refiner_path) if args.refiner_path is not None else get_general_refiner()
    ckpt_dir = os.path.dirname(ckpt_path)

    # --- GT-calibration for crop-coverage stats (both datasets) ---
    gt_calib_stats: dict = {}
    calib_txt_stats = os.path.join(ROOT_DIR, "data", _DS_FOLDER, "pairs_calibrated.txt")
    if os.path.exists(calib_txt_stats):
        if args.dataset == "megadepth":
            gt_calib_stats = load_megadepth_calib(calib_txt_stats)
        else:  # scannet — 4×4 T matrix format
            gt_calib_stats = load_scannet_calib(calib_txt_stats)
        print(f"[STATS] GT calibration loaded for {len(gt_calib_stats)} pairs (crop-coverage analysis)")
    else:
        print("[STATS] pairs_calibrated.txt not found — skipping crop-coverage stats")

    # Accumulators for keypoint-count and crop-coverage stats
    kpt_counts: list = []           # matched kpt count per pair
    epi_dists_all: list = []        # epipolar distances for all base predictions (MegaDepth)
    n_epi_pairs: int = 0            # pairs where we could compute epi stats

    all_mags = []

    with h5py.File(H5_IN, 'r') as f_in:
        pairs = [(n0, n1) for n0 in f_in.keys() for n1 in f_in[n0].keys() if 'matches0' in f_in[n0][n1]]

        for name0, name1 in tqdm(pairs, desc=f"Refinement ({args.dataset}/{args.matcher})"):
            grp_in = f_in[name0][name1]

            if 'matches0' not in grp_in:
                _passthrough(out_files, name0, name1, grp_in)
                continue

            img0 = load_and_resize_to_baseline(key_to_image_path(name0))
            img1 = load_and_resize_to_baseline(key_to_image_path(name1))

            if img0 is None or img1 is None:
                _passthrough(out_files, name0, name1, grp_in)
                continue

            kp0, kp1, m0 = grp_in['keypoints0'][:], grp_in['keypoints1'][:], grp_in['matches0'][:]
            idx0 = np.where(m0 > -1)[0]
            matched_kp0, matched_kp1 = kp0[idx0], kp1[m0[idx0]]

            if len(matched_kp0) == 0:
                _passthrough(out_files, name0, name1, grp_in)
                continue

            # --- Keypoint count & crop-coverage stats ---
            kpt_counts.append(len(matched_kp0))

            if gt_calib_stats and (name0, name1) in gt_calib_stats:
                gt = gt_calib_stats[(name0, name1)]
                F_gt = _compute_F_from_gt(gt['K0'], gt['K1'], gt['R'], gt['t'])
                epi_d = _epi_dist(F_gt,
                                   matched_kp0.astype(np.float64),
                                   matched_kp1.astype(np.float64))
                epi_dists_all.append(epi_d)
                n_epi_pairs += 1

            # Frame-mismatch diagnostic: compare cv2-resized image shape against
            # the H5 keypoint extents. A consistent ratio mismatch (esp. in the
            # short axis) confirms a GF-vs-cv2 resize convention disagreement
            # which manifests as a uniform shift in the visualization.
            if args.frame_diag_max > 0:
                args.frame_diag_max -= 1
                k0_max = kp0.max(axis=0) if len(kp0) else np.array([0, 0])
                k1_max = kp1.max(axis=0) if len(kp1) else np.array([0, 0])
                k0_min = kp0.min(axis=0) if len(kp0) else np.array([0, 0])
                k1_min = kp1.min(axis=0) if len(kp1) else np.array([0, 0])
                print(f"\n[FRAME_DIAG] {name0} -> {name1}")
                print(f"   img0.shape (H,W) = {img0.shape[:2]}   "
                      f"kpts0 x[{k0_min[0]:.1f},{k0_max[0]:.1f}]   "
                      f"kpts0 y[{k0_min[1]:.1f},{k0_max[1]:.1f}]")
                print(f"   img1.shape (H,W) = {img1.shape[:2]}   "
                      f"kpts1 x[{k1_min[0]:.1f},{k1_max[0]:.1f}]   "
                      f"kpts1 y[{k1_min[1]:.1f},{k1_max[1]:.1f}]")
                # Flag suspicious cases: kpts overflow image bounds, or extents
                # reach a value far below the image size (=> kpts in a smaller frame)
                for tag, img, kmax in (("img0", img0, k0_max), ("img1", img1, k1_max)):
                    H_, W_ = img.shape[:2]
                    if kmax[0] > W_ + 1 or kmax[1] > H_ + 1:
                        print(f"   !! {tag}: kpts EXCEED image bounds "
                              f"(kx={kmax[0]:.1f} > W={W_}  or  ky={kmax[1]:.1f} > H={H_})")
                    elif (W_ - kmax[0] > 32) and (H_ - kmax[1] > 32):
                        print(f"   ?? {tag}: kpts span much smaller than image "
                              f"(W={W_}, kpts max x={kmax[0]:.1f};  "
                              f"H={H_}, kpts max y={kmax[1]:.1f}) "
                              f"— possible bucket/EXIF mismatch")

            search_size = 128 if args.num_iters == 1 else 192
            all_deltas = []

            for i in range(0, len(matched_kp0), args.batch_size):
                end = min(i + args.batch_size, len(matched_kp0))

                tensors_t = batch_crop_gpu(img0, matched_kp0[i:end], 128)
                tensors_s = batch_crop_gpu(img1, matched_kp1[i:end], search_size)

                with torch.no_grad():
                    d_native = refiner(tensors_t, tensors_s, num_iters=args.num_iters)
                    d_scaled = d_native[:, :2].cpu().numpy() * scale_factor
                    all_deltas.append(d_scaled)

            deltas = np.concatenate(all_deltas, axis=0)
            all_mags.append(np.linalg.norm(deltas, axis=1))

            # --- GATING LOGIC ---
            for th, f_out in out_files.items():
                current_kp1 = kp1.copy()
                current_matched = matched_kp1.copy()

                mags = np.linalg.norm(deltas, axis=1)
                # Ensure delta is within the maximum threshold (th) AND strictly greater than the inner deadzone
                valid = (mags <= th) & (mags > args.inner_gate)

                current_matched[valid, 0] = matched_kp1[valid, 0] + deltas[valid, 0]
                current_matched[valid, 1] = matched_kp1[valid, 1] + deltas[valid, 1]

                current_kp1[m0[idx0]] = current_matched

                g = f_out.require_group(name0).create_group(name1)
                for k in grp_in.keys():
                    data = current_kp1 if k == 'keypoints1' else grp_in[k][:]
                    g.create_dataset(k, data=data)

    for th, f in out_files.items():
        f.close()
        tmp_p, final_p = final_paths[th]
        os.replace(tmp_p, final_p)

    # ---- Standalone GF eval + H5 cleanup ----
    if not args.skip_eval:
        for th in THRESHOLDS:
            label = _gate_label(th,args.inner_gate) if len(THRESHOLDS) > 1 else f"{args.gate_label}_ig{args.inner_gate:.1f}".rstrip('0').rstrip('.')
            run_gf_eval_and_log(args.dataset, args.matcher, model_name, label, ckpt_dir)
        for th in THRESHOLDS:
            label = _gate_label(th,args.inner_gate) if len(THRESHOLDS) > 1 else f"{args.gate_label}_ig{args.inner_gate:.1f}".rstrip('0').rstrip('.')
            h5_p = os.path.join(PRED_DIR, f"predictions_{model_name}_{label}.h5")
            if os.path.exists(h5_p):
                os.remove(h5_p)

    # ---- |Δ| distribution report ----
    if all_mags:
        mags = np.concatenate(all_mags, axis=0)
        n = len(mags)
        pcts = [50, 75, 90, 95, 99]
        pvals = np.percentile(mags, pcts)
        print("\n[|Δ| distribution over {} correspondences]".format(n))
        print(f"  mean={mags.mean():.3f}px  max={mags.max():.3f}px")
        print(f"  inner_gate={args.inner_gate}px (ignored if <= this)")
        for p, v in zip(pcts, pvals):
            print(f"  p{p:>2}={v:.3f}px")

        print("  gate pass-rate (inner_gate < |Δ| <= th, i.e. refiner APPLIED):")
        for th in [1.0, 2.0, 5.0, 10.0, 20.0, float("inf")]:
            if np.isfinite(th):
                # How many points successfully passed both the outer and inner gates
                pass_mask = (mags <= th) & (mags > args.inner_gate)
                frac = float(pass_mask.mean())
                count = int(pass_mask.sum())
                print(f"    th={th:>4.1f}px : {frac*100:6.2f}%  ({count}/{n})")
            else:
                pass_mask = (mags > args.inner_gate)
                frac = float(pass_mask.mean())
                count = int(pass_mask.sum())
                print(f"    th= inf   : {frac*100:6.2f}%  ({count}/{n})")

    # ---- Keypoint-count report ----
    if kpt_counts:
        kc = np.array(kpt_counts)
        print(f"\n[Matched kpts per pair — {len(kc)} pairs, dataset={args.dataset}]")
        print(f"  mean={kc.mean():.1f}  median={int(np.median(kc))}  "
              f"min={kc.min()}  max={kc.max()}  total={kc.sum()}")
        for pct in [25, 50, 75, 90, 95]:
            print(f"  p{pct}={int(np.percentile(kc, pct))}")

    # ---- Crop-coverage report ----
    if epi_dists_all:
        all_epi = np.concatenate(epi_dists_all)
        n = len(all_epi)
        pcts = [50, 75, 90, 95, 99]
        pvals = np.percentile(all_epi, pcts)

        cov_lines = [
            f"\n[Crop-coverage analysis — {n_epi_pairs} pairs, {n} base predictions, dataset={args.dataset}]",
            f"  Epipolar distance = distance from base-matcher kp1 to the GT epipolar line.",
            f"  If epi_dist > 64px the true match is CONFIRMED outside the 128×128 search crop.",
            f"  (If epi_dist ≤ 64px the true match may still be >64px away along the line.)",
        ]
        for thresh in [5, 10, 32, 64, 128, 256]:
            n_out = int((all_epi > thresh).sum())
            frac  = n_out / n * 100
            cov_lines.append(f"  epi_dist > {thresh:>3d}px : {n_out:>7d} / {n} = {frac:5.1f}%")
        cov_lines.append(
            f"  epi_dist distribution:  mean={all_epi.mean():.1f}px  "
            + "  ".join(f"p{p}={v:.1f}" for p, v in zip(pcts, pvals))
        )
        for ln in cov_lines:
            print(ln)

        # Write to a per-matcher stats file next to the results logs
        matcher_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.matcher)
        # Use the first threshold label as the file suffix (matches results_*.txt naming)
        _first_label = (_gate_label(THRESHOLDS[0], args.inner_gate)
                        if len(THRESHOLDS) > 1
                        else f"{args.gate_label}_ig{args.inner_gate:.1f}".rstrip('0').rstrip('.'))
        _stats_path = os.path.join(
            ckpt_dir,
            f"crop_coverage_{args.dataset}_{matcher_tag}_{_first_label}.txt"
        )
        with open(_stats_path, 'w', encoding='utf-8') as _sf:
            _sf.write("\n".join(cov_lines) + "\n")
        print(f"[Crop-coverage] stats saved to {_stats_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()