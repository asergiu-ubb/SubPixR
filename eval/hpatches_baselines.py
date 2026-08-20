import argparse
import csv
import datetime
import glob
import heapq
import os
import random
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- glue-factory (git clone) instead of pip lightglue ---
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../glue-factory'))

import kornia.feature as KF
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.models.matchers.lightglue import LightGlue

from subpixr.utils import standardize_path, get_general_refiner, load_refiner
from s2dnet_refiner import load_s2dnet, extract_pyramid, refine_hierarchical

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="superpoint-lightglue",
                    choices=["superpoint-nn", "disk-nn",
                             "superpoint-lightglue", "disk-lightglue",
                             "superpoint-superglue", "superpoint-caps",
                             "loftr", "aspanformer", "patch2pix"],
                    help="Which baseline matcher to evaluate.")
parser.add_argument("--hpatches_root", type=str,
                    default="./datasets/hpatches-sequences-release")
parser.add_argument("--max_edge", type=int, default=9999)
parser.add_argument("--refiner_path", type=str, default=None,
                    help="Path to refiner .pth checkpoint. Defaults to get_general_refiner().")
parser.add_argument("--num_iters", type=int, default=1,
                    help="Refiner iterations. 1 → 128px search; >1 → 192px search.")
parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size for refiner forward pass.")
parser.add_argument("--max_correction_px", type=float, default=30.0,
                    help="Gate: skip refiner output if |delta| > this (px).")
parser.add_argument("--cache_dir", type=str, default=None,
                    help="Path to the feature cache written by hpatches_homography_estimation.py ")
parser.add_argument("--force_recompute_matches", action="store_true", default=False,
                    help="Ignore existing match cache and rerun the matcher.")
parser.add_argument("--export_val_dir", type=str, default=None,
                    help="If set, export a RefinementDataset-compatible val set.")
parser.add_argument("--skip_s2dnet", action="store_true", default=True,
                    help="Skip S2DNet hierarchical refiner column.")
args = parser.parse_args()

RUN_S2DNET = not args.skip_s2dnet

_LOFTR_INT_MAX = 1024# 1200
_P2P_INT_MAX = 1024
# ---------------------------------------------------------------------------
# Global Flags
# ---------------------------------------------------------------------------
EXPORT_VAL_HPATCHES = args.export_val_dir is not None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_edge_tag = f"e{args.max_edge}" if args.max_edge > 0 else "full"
CACHE_DIR = args.cache_dir
if CACHE_DIR is None:
    _auto = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"cache_hpatches_superpoint_{_edge_tag}")
    if args.method.startswith("superpoint") and os.path.isdir(_auto):
        CACHE_DIR = _auto
        print(f"[cache] Auto-detected SuperPoint cache: {CACHE_DIR}")

USE_CACHE = (CACHE_DIR is not None) and args.method.startswith("superpoint")

# Match cache
_script_dir = os.path.dirname(os.path.abspath(__file__))
MATCHES_CACHE_DIR = os.path.join(_script_dir, f"cache_hpatches_matches_{args.method}_{_edge_tag}")
FORCE_RECOMPUTE_MATCHES = args.force_recompute_matches
_n_match_cached = sum(1 for _ in __import__('pathlib').Path(MATCHES_CACHE_DIR).rglob("*.npz")) \
    if os.path.isdir(MATCHES_CACHE_DIR) else 0
_matches_fully_cached = (not FORCE_RECOMPUTE_MATCHES) and (_n_match_cached >= 570)

if _matches_fully_cached:
    print(f"[match-cache] Fully cached ({_n_match_cached} files) — skipping SOTA model loading.")
elif _n_match_cached > 0:
    print(f"[match-cache] Partial cache ({_n_match_cached}/580 files) — will compute missing pairs.")
else:
    print(f"[match-cache] Will create on first run: {MATCHES_CACHE_DIR}")

# ---------------------------------------------------------------------------
# --- 1. INITIALIZE MODELS ---
# ---------------------------------------------------------------------------
sp_model = disk_extractor = lightglue_matcher = loftr_matcher = None
sg_sp_model = sg_matcher = aspanformer_matcher = patch2pix_model = caps_model = None

if not _matches_fully_cached and args.method.startswith("superpoint"):
    if not USE_CACHE:
        sp_model = SuperPoint({'max_num_keypoints': 2048, 'nms_radius': 4,
                               'detection_threshold': 0.005}).eval().to(DEVICE)
    if args.method == "superpoint-lightglue":
        lightglue_matcher = LightGlue({'weights': 'superpoint', 'input_dim': 256}).eval().to(DEVICE)

# --- CAPS INITIALIZATION ---
if not _matches_fully_cached and args.method == "superpoint-caps":
    _caps_root = os.path.abspath('../caps')
    _caps_ckpt = os.path.join(_caps_root, 'weights', 'caps-pretrained.pth')
    try:
        _saved_path = sys.path[:]
        sys.path.insert(0, os.path.join(_caps_root, 'CAPS'))
        from network import CAPSNet

        sys.path = _saved_path
        import types

        _caps_args = types.SimpleNamespace(
            pretrained=1, backbone='resnet50', coarse_feat_dim=128, fine_feat_dim=128,
            prob_from='correlation', use_nn=1, window_size=0.125,
        )
        caps_model = CAPSNet(_caps_args, DEVICE).eval().to(DEVICE)
        _ckpt = torch.load(_caps_ckpt, map_location=DEVICE)
        caps_model.load_state_dict(_ckpt.get("state_dict", _ckpt))
    except ImportError as e:
        raise RuntimeError(f"CAPS import failed ({e}).")

elif not _matches_fully_cached and args.method in ("disk-nn", "disk-lightglue"):
    disk_extractor = KF.DISK.from_pretrained('depth').eval().to(DEVICE)
    if args.method == "disk-lightglue":
        lightglue_matcher = LightGlue({'weights': 'disk', 'input_dim': 128}).eval().to(DEVICE)

elif not _matches_fully_cached and args.method == "superpoint-superglue":
    try:
        sys.path.insert(0, os.path.abspath('../SuperGluePretrainedNetwork'))
        from models.superpoint import SuperPoint as SGSuperPoint
        from models.superglue import SuperGlue

        sg_sp_model = SGSuperPoint({'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 2048}).eval().to(
            DEVICE)
        sg_matcher = SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(
            DEVICE)
    except ImportError:
        raise RuntimeError("SuperGlue repo not found.")

elif not _matches_fully_cached and args.method == "loftr":
    loftr_matcher = KF.LoFTR(pretrained='outdoor').eval().to(DEVICE)

elif not _matches_fully_cached and args.method == "aspanformer":
    _aspan_root = os.path.abspath('../ml-aspanformer')
    _aspan_ckpt = os.path.join(_aspan_root, 'weights', 'outdoor.ckpt')
    try:
        _saved_path = sys.path[:]
        sys.path = [_aspan_root] + [p for p in sys.path if
                                    p not in ('', '.', '..', os.path.abspath('.'), os.path.abspath('..'))]
        from src.ASpanFormer.aspanformer import ASpanFormer as ASpanFormerModel
        from src.config.default import get_cfg_defaults

        sys.path = _saved_path
        cfg = get_cfg_defaults()
        # Match the official configs/aspan/outdoor/aspan_test.py so online_resize
        # has valid train/test res (defaults are None, which breaks resize_input).
        cfg.ASPAN.COARSE.TRAIN_RES = [832, 832]
        cfg.ASPAN.COARSE.TEST_RES = [1152, 1152]


        def _lower_cfg(node):
            from yacs.config import CfgNode as _CN
            return {k.lower(): (_lower_cfg(v) if isinstance(v, _CN) else v) for k, v in node.items()}


        aspanformer_matcher = ASpanFormerModel(config=_lower_cfg(cfg['ASPAN'])).eval().to(DEVICE)
        weights = torch.load(_aspan_ckpt, map_location=DEVICE)
        aspanformer_matcher.load_state_dict(weights['state_dict'], strict=False)
    except ImportError as e:
        raise RuntimeError(f"ASpanFormer import failed ({e}).")

elif not _matches_fully_cached and args.method == "patch2pix":
    _p2p_root = os.path.abspath('../patch2pix')
    _p2p_ckpt = os.path.join(_p2p_root, 'pretrained', 'patch2pix_pretrained.pth')
    try:
        _saved_path = sys.path[:]
        sys.path = [_p2p_root] + [p for p in sys.path if
                                  p not in ('', '.', '..', os.path.abspath('.'), os.path.abspath('..'))]
        from utils.eval.model_helper import load_model as p2p_load_model
        from utils.eval.model_helper import estimate_matches as p2p_estimate

        sys.path = _saved_path
        patch2pix_model = p2p_load_model(_p2p_ckpt, method='patch2pix')
    except ImportError as e:
        raise RuntimeError(f"Patch2Pix import failed ({e}).")

# ---------------------------------------------------------------------------
# --- 1b. LOAD REFINER ---
# ---------------------------------------------------------------------------
_refiner_weights = args.refiner_path if args.refiner_path else get_general_refiner()
REFINER_SEARCH_SIZE = 128 if args.num_iters == 1 else 192

try:
    refiner_model, REFINER_SCALE_FACTOR = load_refiner(_refiner_weights, device=DEVICE)
    print(f"[refiner] Loaded: {_refiner_weights}  scale_factor={REFINER_SCALE_FACTOR}")
except Exception as e:
    print(f"[SKIP] Cannot load refiner {_refiner_weights}: {e}", file=sys.stderr)
    sys.exit(2)

# --- S2DNet (hierarchical sparse-to-dense refiner, same pipeline as
# eval_S2DNet_sota.py on MegaDepth/ScanNet). Applied to matcher inliers before
# RANSAC, gated by cycle-consistency + max_correction_px. ---
s2dnet_model = None
s2d_level_scales = None
if RUN_S2DNET:
    _S2D_ROOT = os.path.abspath('../S2DNet-Minimal')
    _S2D_CKPT = os.path.join(_S2D_ROOT, 's2dnet_weights.pth')
    try:
        s2dnet_model, s2d_level_scales = load_s2dnet(_S2D_CKPT, DEVICE)
        print(f"[S2DNet] Loaded: {_S2D_CKPT}  strides={s2d_level_scales}")
    except Exception as _s2d_err:
        print(f"[S2DNet] Skipped: {_s2d_err}")
else:
    print("[S2DNet] Skipped via --skip_s2dnet.")


# ---------------------------------------------------------------------------
# --- 2. HELPERS ---
# ---------------------------------------------------------------------------
def resize_to_max_edge(img_rgb: np.ndarray, max_edge: int):
    h, w = img_rgb.shape[:2]
    scale = max_edge / max(h, w)
    if scale >= 1.0:
        return img_rgb, 1.0
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h)), scale


def compute_corner_error(H_gt, H_est, img_shape):
    if H_est is None or H_gt is None:
        return np.inf
    h, w = img_shape[:2]
    corners = np.array([[0, 0, 1], [0, h - 1, 1], [w - 1, 0, 1], [w - 1, h - 1, 1]]).T
    warped_gt = H_gt @ corners;
    warped_gt = warped_gt[:2] / warped_gt[2]
    warped_est = H_est @ corners;
    warped_est = warped_est[:2] / warped_est[2]
    return np.mean(np.linalg.norm(warped_gt - warped_est, axis=0))


_NORM_MEAN_T = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_NORM_STD_T = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def _image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """Converts HWC numpy image to 1CHW normalized GPU tensor."""
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    return (t - _NORM_MEAN_T) / _NORM_STD_T


def refine_matches_gpu(pts1: np.ndarray, pts2: np.ndarray,
                       ref_tensor: torch.Tensor, trg_tensor: torch.Tensor) -> np.ndarray:
    """GPU-accelerated refinement directly from full-image tensors."""
    pts2_refined = pts2.copy()
    N = len(pts1)
    if N == 0: return pts2_refined

    pts1_t = torch.from_numpy(pts1).float().to(DEVICE)
    pts2_t = torch.from_numpy(pts2).float().to(DEVICE)

    with torch.no_grad():
        for b in range(0, N, args.batch_size):
            end = min(b + args.batch_size, N)
            b_size = end - b

            b_pts1 = pts1_t[b:end]
            b_pts2 = pts2_t[b:end]

            # Expand full images to batch dimension to enable parallel grid_sample
            b_ref = ref_tensor.expand(b_size, -1, -1, -1)
            b_trg = trg_tensor.expand(b_size, -1, -1, -1)

            # Match CPU snapping logic exactly: round coords, then subtract half-size
            cx_1 = torch.round(b_pts1[:, 0])
            cy_1 = torch.round(b_pts1[:, 1])
            tl_1 = torch.stack([cx_1 - 64.0, cy_1 - 64.0], dim=1)

            cx_2 = torch.round(b_pts2[:, 0])
            cy_2 = torch.round(b_pts2[:, 1])
            tl_2 = torch.stack([cx_2 - (REFINER_SEARCH_SIZE / 2.0), cy_2 - (REFINER_SEARCH_SIZE / 2.0)], dim=1)

            # Parallel GPU Crop
            t_crops = refiner_model.extract_patch(b_ref, tl_1, 128)
            s_crops = refiner_model.extract_patch(b_trg, tl_2, REFINER_SEARCH_SIZE)

            # Predict
            preds = refiner_model(t_crops, s_crops, num_iters=args.num_iters)
            deltas = preds.cpu().numpy() * REFINER_SCALE_FACTOR

            for k in range(b_size):
                dx, dy = float(deltas[k, 0]), float(deltas[k, 1])
                if (dx ** 2 + dy ** 2) ** 0.5 <= args.max_correction_px:
                    snap_x = float(round(pts2[b + k][0]))
                    snap_y = float(round(pts2[b + k][1]))
                    pts2_refined[b + k] = [snap_x + dx, snap_y + dy]

    return pts2_refined


def refine_matches_s2dnet(pts1: np.ndarray, pts2: np.ndarray,
                          ref_pyr: list, trg_pyr: list,
                          level_scales: tuple,
                          max_corr_px: float) -> np.ndarray:
    """S2DNet hierarchical coarse-to-fine refinement of matched points.

    Points failing cycle-consistency or whose |delta| exceeds max_corr_px are
    left at their baseline kp1 — no refinement, no forced fallback to RANSAC
    outlier territory.
    """
    pts2_refined = pts2.copy()
    if len(pts1) == 0:
        return pts2_refined
    deltas, cycle_ok = refine_hierarchical(
        ref_pyr, trg_pyr,
        pts1.astype(np.float32), pts2.astype(np.float32),
        level_scales,
    )
    mags = np.linalg.norm(deltas, axis=1)
    valid = cycle_ok & (mags <= max_corr_px)
    pts2_refined[valid] = pts2[valid] + deltas[valid]
    return pts2_refined


def compute_auc(errors, thresholds=(1, 3, 5, 10)):
    errors = np.sort(np.array(errors))
    aucs = []
    for t in thresholds:
        valid = errors[errors <= t]
        if len(valid) == 0:
            aucs.append(0.0)
            continue
        recall = np.arange(1, len(valid) + 1) / len(errors)
        aucs.append(np.trapz(np.append([0], recall), np.append([0], valid)) / t)
    return aucs


def load_cached_features(cache_path: str):
    data = np.load(cache_path)
    return data['kp'], data['des']


def feats_from_cache(kp_np: np.ndarray, des_np: np.ndarray, img_hw):
    h, w = img_hw
    kp = torch.from_numpy(kp_np)[None].to(DEVICE)
    des = torch.from_numpy(des_np)[None].to(DEVICE)
    sz = torch.tensor([[w, h]], dtype=torch.float32, device=DEVICE)
    return kp, des, sz


def lightglue_data(kp0, des0, sz0, kp1, des1, sz1):
    return {
        'keypoints0': kp0, 'keypoints1': kp1,
        'descriptors0': des0, 'descriptors1': des1,
        'view0': {'image_size': sz0}, 'view1': {'image_size': sz1},
    }


def parse_lightglue_matches(out, kp0, kp1):
    if 'matches0' in out:
        m0 = out['matches0'][0]
        valid = m0 > -1
        pts1 = kp0[0][valid].cpu().numpy()
        pts2 = kp1[0][m0[valid]].cpu().numpy()
    elif 'matches' in out:
        matches = out['matches']
        if isinstance(matches, list): matches = matches[0]
        if len(matches) == 0: return np.empty((0, 2)), np.empty((0, 2))
        pts1 = kp0[0][matches[:, 0]].cpu().numpy()
        pts2 = kp1[0][matches[:, 1]].cpu().numpy()
    else:
        return np.empty((0, 2)), np.empty((0, 2))
    return pts1, pts2


def extract_superpoint(img_rgb: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    t = torch.from_numpy(gray).float()[None, None] / 255.0
    with torch.no_grad(): out = sp_model({'image': t.to(DEVICE)})
    kp, des = out['keypoints'][0], out['descriptors'][0]
    if des.shape[0] == 256: des = des.T
    h, w = img_rgb.shape[:2]
    sz = torch.tensor([[w, h]], dtype=torch.float32, device=DEVICE)
    return kp[None], des[None], sz


def get_matches(img1_rgb, img2_rgb, method, cache1=None, cache2=None, img1_hw=None, img2_hw=None):
    with torch.no_grad():
        if method == "loftr":
            # LoFTR builds an (HW/64)² coarse sim matrix → blows past 80 GB at native HPatches res.
            # Internally cap the longer edge to _LOFTR_INT_MAX, then scale matches back to input res.

            h1, w1 = img1_rgb.shape[:2]; h2, w2 = img2_rgb.shape[:2]
            s1 = min(1.0, _LOFTR_INT_MAX / max(h1, w1))
            s2 = min(1.0, _LOFTR_INT_MAX / max(h2, w2))
            # Round to /8 (LoFTR coarse stride) to avoid shape/mask edge cases.
            def _r8(x): return max(8, (x // 8) * 8)
            nw1, nh1 = _r8(int(round(w1 * s1))), _r8(int(round(h1 * s1)))
            nw2, nh2 = _r8(int(round(w2 * s2))), _r8(int(round(h2 * s2)))
            g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
            if (nh1, nw1) != (h1, w1): g1 = cv2.resize(g1, (nw1, nh1))
            if (nh2, nw2) != (h2, w2): g2 = cv2.resize(g2, (nw2, nh2))
            t1 = torch.from_numpy(g1).float()[None, None] / 255.0
            t2 = torch.from_numpy(g2).float()[None, None] / 255.0
            out = loftr_matcher({'image0': t1.to(DEVICE), 'image1': t2.to(DEVICE)})
            kp1 = out['keypoints0'].cpu().numpy().astype(np.float32)
            kp2 = out['keypoints1'].cpu().numpy().astype(np.float32)
            # Rescale matches from internal LoFTR resolution back to original input resolution.
            if (nh1, nw1) != (h1, w1):
                kp1[:, 0] *= (w1 / nw1); kp1[:, 1] *= (h1 / nh1)
            if (nh2, nw2) != (h2, w2):
                kp2[:, 0] *= (w2 / nw2); kp2[:, 1] *= (h2 / nh2)
            return kp1, kp2

        if method in ("superpoint-nn", "superpoint-lightglue", "superpoint-caps"):
            cache_hit = (USE_CACHE and cache1 and cache2 and os.path.exists(cache1) and os.path.exists(cache2))
            if cache_hit:
                kp1_np, des1_np = load_cached_features(cache1)
                kp2_np, des2_np = load_cached_features(cache2)
                kp0_t, des0_t, sz0 = feats_from_cache(kp1_np, des1_np, img1_hw)
                kp1_t, des1_t, sz1 = feats_from_cache(kp2_np, des2_np, img2_hw)
            else:
                if sp_model is None: raise RuntimeError("SuperPoint model not initialised.")
                kp0_t, des0_t, sz0 = extract_superpoint(img1_rgb)
                kp1_t, des1_t, sz1 = extract_superpoint(img2_rgb)

            if method == "superpoint-lightglue":
                out = lightglue_matcher(lightglue_data(kp0_t, des0_t, sz0, kp1_t, des1_t, sz1))
                return parse_lightglue_matches(out, kp0_t, kp1_t)
            elif method == "superpoint-nn":
                _, idxs = KF.match_nn(des0_t[0], des1_t[0])
                if len(idxs) == 0: return np.empty((0, 2)), np.empty((0, 2))
                return kp0_t[0][idxs[:, 0]].cpu().numpy(), kp1_t[0][idxs[:, 1]].cpu().numpy()
            elif method == "superpoint-caps":
                t1 = torch.from_numpy(img1_rgb).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0
                t2 = torch.from_numpy(img2_rgb).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0
                coord2_ef, std = caps_model.test(t1, t2, kp0_t)
                matched_kp1, matched_kp2 = kp0_t[0].cpu().numpy(), coord2_ef[0].cpu().numpy()
                if len(matched_kp1) == 0: return np.empty((0, 2)), np.empty((0, 2))
                return matched_kp1, matched_kp2

        if method in ("disk-nn", "disk-lightglue"):
            t1 = torch.from_numpy(img1_rgb).float().permute(2, 0, 1)[None] / 255.0
            t2 = torch.from_numpy(img2_rgb).float().permute(2, 0, 1)[None] / 255.0
            out1 = disk_extractor(t1.to(DEVICE), n=2048, pad_if_not_divisible=True)[0]
            out2 = disk_extractor(t2.to(DEVICE), n=2048, pad_if_not_divisible=True)[0]
            h1, w1 = img1_rgb.shape[:2];
            h2, w2 = img2_rgb.shape[:2]
            sz0 = torch.tensor([[w1, h1]], dtype=torch.float32, device=DEVICE)
            sz1 = torch.tensor([[w2, h2]], dtype=torch.float32, device=DEVICE)
            kp0_t, des0_t = out1.keypoints[None], out1.descriptors[None]
            kp1_tt, des1_tt = out2.keypoints[None], out2.descriptors[None]

            if method == "disk-lightglue":
                out = lightglue_matcher(lightglue_data(kp0_t, des0_t, sz0, kp1_tt, des1_tt, sz1))
                return parse_lightglue_matches(out, kp0_t, kp1_tt)
            else:
                _, idxs = KF.match_nn(des0_t[0], des1_tt[0])
                if len(idxs) == 0: return np.empty((0, 2)), np.empty((0, 2))
                return kp0_t[0][idxs[:, 0]].cpu().numpy(), kp1_tt[0][idxs[:, 1]].cpu().numpy()

        if method == "superpoint-superglue":
            gray1 = torch.from_numpy(cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)).float()[None, None] / 255.0
            gray2 = torch.from_numpy(cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)).float()[None, None] / 255.0
            pred0 = sg_sp_model({'image': gray1.to(DEVICE)})
            pred1 = sg_sp_model({'image': gray2.to(DEVICE)})
            data = {
                'keypoints0': torch.stack(pred0['keypoints']), 'scores0': torch.stack(pred0['scores']),
                'descriptors0': torch.stack(pred0['descriptors']),
                'keypoints1': torch.stack(pred1['keypoints']), 'scores1': torch.stack(pred1['scores']),
                'descriptors1': torch.stack(pred1['descriptors']),
                'image0': gray1.to(DEVICE), 'image1': gray2.to(DEVICE),
            }
            out = sg_matcher(data)
            matches = out['matches0'][0]
            valid = matches > -1
            pts1 = pred0['keypoints'][0][valid].cpu().numpy()
            pts2 = pred1['keypoints'][0][matches[valid]].cpu().numpy()
            return pts1, pts2

        if method == "aspanformer":
            # ASpanFormer's COARSEST_LEVEL=[26,26] makes the coarse transformer's
            # pool/upsample require (H/8) % int((H/8)/26) == 0, which fails for many
            # aspect ratios (e.g. 640-wide → ds=3, 80%3≠0, "Expected 80 got 78").
            # online_resize=True forces ds=[4,4] and handles that constraint.
            # Additional hard cap: the sinusoidal position-encoding buffer is sized
            # for long side ≤ 256; feeding a 384-wide image triggers
            # "tensor a (384) must match tensor b (256)" in position_encoding.py.
            # Pre-scale so the long side ≤ _ASP_MAX_LONG (keep df=32 multiples),
            # then scale mkpts back into original-image coords.
            _ASP_MAX_LONG = 256

            def _prep_for_aspan(img_rgb):
                h, w = img_rgb.shape[:2]
                if max(h, w) <= _ASP_MAX_LONG:
                    return img_rgb, (1.0, 1.0)
                s = _ASP_MAX_LONG / float(max(h, w))
                new_w = max(32, int(round(w * s / 32.0)) * 32)
                new_h = max(32, int(round(h * s / 32.0)) * 32)
                resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Scale factors to recover original coords from resized coords.
                return resized, (w / float(new_w), h / float(new_h))

            def _gray_tensor(img_rgb):
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                return torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]

            img1_r, sc1 = _prep_for_aspan(img1_rgb)
            img2_r, sc2 = _prep_for_aspan(img2_rgb)
            data = {'image0': _gray_tensor(img1_r).to(DEVICE),
                    'image1': _gray_tensor(img2_r).to(DEVICE)}
            aspanformer_matcher(data, online_resize=True)
            m0 = data['mkpts0_f'].cpu().numpy()
            m1 = data['mkpts1_f'].cpu().numpy()
            if len(m0):
                m0 = m0 * np.array([sc1[0], sc1[1]], dtype=np.float32)
                m1 = m1 * np.array([sc2[0], sc2[1]], dtype=np.float32)
            return m0, m1

        if method == "patch2pix":
            # Patch2Pix maxpool4d is O((H/16·W/16)² · k⁴) → OOMs at native res.
            # Force its internal resize via imsize= (longer edge); matches come back in orig coords.

            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f1, tempfile.NamedTemporaryFile(
                    suffix='.jpg', delete=False) as f2:
                cv2.imwrite(f1.name, cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f2.name, cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
                tmp1, tmp2 = f1.name, f2.name
            try:
                matches, _, _ = p2p_estimate(patch2pix_model, tmp1, tmp2, ksize=2, io_thres=0.25, eval_type='fine',
                                             imsize=_P2P_INT_MAX)
            finally:
                os.unlink(tmp1);
                os.unlink(tmp2)
            if len(matches) == 0: return np.empty((0, 2)), np.empty((0, 2))
            return matches[:, :2].astype(np.float32), matches[:, 2:4].astype(np.float32)

    return np.empty((0, 2)), np.empty((0, 2))


# ---------------------------------------------------------------------------
# --- 3. MAIN EVALUATION ---
# ---------------------------------------------------------------------------

EXPORT_CAP = 4000
_export_reservoir = []
_export_count = 0
_export_csv_f = _export_csv_w = None

if EXPORT_VAL_HPATCHES:
    os.makedirs(args.export_val_dir, exist_ok=True)
    _export_csv_path = os.path.join(args.export_val_dir, "labels.csv")
    _export_csv_f = open(_export_csv_path, "w", newline="")
    _export_csv_w = csv.writer(_export_csv_f)
    _export_csv_w.writerow(["filename_template", "filename_search", "dx", "dy", "category"])
    print(f"[export] Val HPatches → {args.export_val_dir}  (cap={EXPORT_CAP})")

sequences = sorted(glob.glob(os.path.join(args.hpatches_root, "*")))
errors = {'i': [], 'v': [], 'all': []}
refined_errors = {'i': [], 'v': [], 'all': []}
s2dnet_errors = {'i': [], 'v': [], 'all': []}

CKPT_DIR = os.path.dirname(_refiner_weights)


# Helper function for CPU cropping (only used if EXPORT/DEBUG is True)
def _crop_np(img_rgb: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    x1, y1 = int(round(cx - size / 2)), int(round(cy - size / 2))
    h, w = img_rgb.shape[:2]
    pad_t, pad_b = max(0, -y1), max(0, (y1 + size) - h)
    pad_l, pad_r = max(0, -x1), max(0, (x1 + size) - w)
    if pad_t or pad_b or pad_l or pad_r:
        img_rgb = cv2.copyMakeBorder(img_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        x1 += pad_l;
        y1 += pad_t
    patch = img_rgb[y1:y1 + size, x1:x1 + size]
    if patch.shape[0] != size or patch.shape[1] != size: patch = cv2.resize(patch, (size, size))
    return patch


for seq in tqdm(sequences, desc=f"Eval {args.method}"):
    seq_type, seq_name = os.path.basename(seq)[0], os.path.basename(seq)

    ref_img = cv2.cvtColor(cv2.imread(os.path.join(seq, "1.ppm")), cv2.COLOR_BGR2RGB)
    ref_img, scale_ref = resize_to_max_edge(ref_img, args.max_edge)
    img_shape = ref_img.shape

    # Pre-convert to tensor for fast GPU extraction
    ref_tensor = _image_to_tensor(ref_img)
    # S2DNet ref pyramid is amortised across all 5 target pairs of this seq.
    ref_pyr = extract_pyramid(ref_img, s2dnet_model, DEVICE) if s2dnet_model is not None else None

    for i in range(2, 7):
        trg_path, h_path = os.path.join(seq, f"{i}.ppm"), os.path.join(seq, f"H_1_{i}")
        if not os.path.exists(trg_path) or not os.path.exists(h_path): continue

        trg_img = cv2.cvtColor(cv2.imread(trg_path), cv2.COLOR_BGR2RGB)
        trg_img, scale_trg = resize_to_max_edge(trg_img, args.max_edge)
        trg_tensor = _image_to_tensor(trg_img)
        trg_pyr = extract_pyramid(trg_img, s2dnet_model, DEVICE) if s2dnet_model is not None else None

        H_gt_raw = np.loadtxt(h_path)
        S_ref, S_trg = np.diag([scale_ref, scale_ref, 1.0]), np.diag([scale_trg, scale_trg, 1.0])
        H_gt = S_trg @ H_gt_raw @ np.linalg.inv(S_ref)

        cache_ref = os.path.join(CACHE_DIR, seq_name, "1.npz") if CACHE_DIR else None
        cache_trg = os.path.join(CACHE_DIR, seq_name, f"{i}.npz") if CACHE_DIR else None

        _match_cache_path = os.path.join(MATCHES_CACHE_DIR, seq_name, f"{i}.npz")
        if not FORCE_RECOMPUTE_MATCHES and os.path.exists(_match_cache_path):
            _mc = np.load(_match_cache_path)
            pts1, pts2 = _mc["pts1"], _mc["pts2"]
        else:
            pts1, pts2 = get_matches(ref_img, trg_img, args.method, cache1=cache_ref, cache2=cache_trg,
                                     img1_hw=ref_img.shape[:2], img2_hw=trg_img.shape[:2])
            os.makedirs(os.path.dirname(_match_cache_path), exist_ok=True)
            np.savez_compressed(_match_cache_path, pts1=pts1, pts2=pts2)

        if len(pts1) < 4:
            err = err_ref = err_s2d = np.inf
        else:
            H_est, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            err = compute_corner_error(H_gt, H_est, img_shape)

            if mask is not None and int(mask.sum()) >= 4:
                inlier_idx = np.where(mask.ravel() == 1)[0]
                pts1_in, pts2_in = pts1[inlier_idx], pts2[inlier_idx]

                # --- FAST GPU REFINEMENT ---
                pts2_ref = refine_matches_gpu(pts1_in, pts2_in, ref_tensor, trg_tensor)
                H_ref, _ = cv2.findHomography(pts1_in, pts2_ref, cv2.RANSAC, 3.0)
                err_ref = compute_corner_error(H_gt, H_ref, img_shape)

                # --- S2DNet HIERARCHICAL REFINEMENT (parallel column) ---
                if s2dnet_model is not None:
                    pts2_s2d = refine_matches_s2dnet(
                        pts1_in, pts2_in, ref_pyr, trg_pyr,
                        s2d_level_scales, args.max_correction_px,
                    )
                    H_s2d, _ = cv2.findHomography(pts1_in, pts2_s2d, cv2.RANSAC, 3.0)
                    err_s2d = compute_corner_error(H_gt, H_s2d, img_shape)
                else:
                    err_s2d = np.inf

                # --- OPTIONAL: EXPORT / DEBUG CPU TASKS ---
                # Bypassed automatically during speed-run evaluation
                if EXPORT_VAL_HPATCHES:
                    deltas = pts2_ref - pts2_in
                    for _k in range(len(pts1_in)):
                        _dx, _dy = float(deltas[_k, 0]), float(deltas[_k, 1])
                        _mag = (_dx ** 2 + _dy ** 2) ** 0.5

                        _t = cv2.cvtColor(_crop_np(ref_img, pts1_in[_k, 0], pts1_in[_k, 1], 128), cv2.COLOR_RGB2BGR)
                        _s = cv2.cvtColor(_crop_np(trg_img, pts2_in[_k, 0], pts2_in[_k, 1], REFINER_SEARCH_SIZE),
                                          cv2.COLOR_RGB2BGR)

                        _pt1h = np.array([pts1_in[_k, 0], pts1_in[_k, 1], 1.0])
                        _pt2gt = H_gt @ _pt1h;
                        _pt2gt = _pt2gt / _pt2gt[2]
                        _gt_dx, _gt_dy = float(_pt2gt[0] - pts2_in[_k, 0]), float(_pt2gt[1] - pts2_in[_k, 1])
                        _entry = (_mag, _t, _s, _dx, _dy, _gt_dx, _gt_dy, seq_name)

                        if EXPORT_VAL_HPATCHES:
                            _export_count += 1
                            _cat = f"{seq_type}_{seq_name}"
                            _ft, _fs = f"patch_{_export_count:06d}_t.jpg", f"patch_{_export_count:06d}_s.jpg"
                            cv2.imwrite(os.path.join(args.export_val_dir, _ft), _t)
                            cv2.imwrite(os.path.join(args.export_val_dir, _fs), _s)
                            _export_reservoir.append((_ft, _fs, _gt_dx, _gt_dy, _cat))

            else:
                err_ref = err_s2d = np.inf

        errors[seq_type].append(err);
        errors['all'].append(err)
        refined_errors[seq_type].append(err_ref);
        refined_errors['all'].append(err_ref)
        s2dnet_errors[seq_type].append(err_s2d);
        s2dnet_errors['all'].append(err_s2d)

if EXPORT_VAL_HPATCHES:
    selected = _export_reservoir
    if len(selected) > EXPORT_CAP:
        selected_set = set(random.sample(range(len(selected)), EXPORT_CAP))
        for _i, (_ft, _fs, *_) in enumerate(_export_reservoir):
            if _i not in selected_set:
                for _p in (_ft, _fs):
                    _full = os.path.join(args.export_val_dir, _p)
                    if os.path.exists(_full): os.remove(_full)
        selected = [_export_reservoir[_i] for _i in sorted(selected_set)]
    for _ft, _fs, _gt_dx, _gt_dy, _cat in selected:
        _export_csv_w.writerow([_ft, _fs, f"{_gt_dx:.4f}", f"{_gt_dy:.4f}", _cat])
    _export_csv_f.close()
    print(f"[export] Wrote {len(selected)} patches → {args.export_val_dir}/labels.csv")

# ---------------------------------------------------------------------------
# --- 4. REPORT + LOG ---
# ---------------------------------------------------------------------------
thresholds = [1, 3, 5, 10]

lines = []
lines.append("\n" + "=" * 65)
lines.append(f" Baseline: {args.method.upper()} | Max Edge: {args.max_edge}px")
lines.append(f" Cache    : {CACHE_DIR if USE_CACHE else 'disabled (fresh extraction)'}")
lines.append(f" Refiner  : {_refiner_weights}  scale={REFINER_SCALE_FACTOR}")
lines.append(f" Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 65)

col_w = 10
_s2d_on = s2dnet_model is not None
hdr = f"{'Category':<14} | {'Thresh':>6} | {'Baseline':>{col_w}} | {'+ Refiner':>{col_w}}"
if _s2d_on:
    hdr += f" | {'+ S2DNet':>{col_w}}"
lines.append(hdr);
lines.append("-" * len(hdr))

for category, label in zip(['all', 'i', 'v'], ['Overall', 'Illumination', 'Viewpoint']):
    aucs_base = compute_auc(errors[category], thresholds)
    aucs_ref = compute_auc(refined_errors[category], thresholds)
    aucs_s2d = compute_auc(s2dnet_errors[category], thresholds) if _s2d_on else None
    for idx, t in enumerate(thresholds):
        cat_str = label if idx == 0 else ""
        row = (f"{cat_str:<14} | {t:>5}px | {aucs_base[idx] * 100:>{col_w}.2f}% "
               f"| {aucs_ref[idx] * 100:>{col_w}.2f}%")
        if _s2d_on:
            row += f" | {aucs_s2d[idx] * 100:>{col_w}.2f}%"
        lines.append(row)
    lines.append("-" * len(hdr))

lines.append("=" * 65);
output = "\n".join(lines)
print(output)

if os.environ.get("HPATCHES_SKIP_LOG", "0") == "1":
    print("[log] HPATCHES_SKIP_LOG=1 → skipping per-method log file write.")
else:
    log_path = os.path.join(CKPT_DIR, f"hpatches_strategyA_{args.method}_e{args.max_edge}.txt")
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as _lf:
        _lf.write(output + "\n")
    print(f"Log saved to: {log_path}")
