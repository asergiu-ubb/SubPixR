import argparse
import contextlib
import glob
import heapq
import io
import json
import os
import random
import sys
import tempfile

import cv2
import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_THIS_DIR, '..'))          # eval_sota_with_refiner/
sys.path.append(os.path.join(_THIS_DIR, '..', '..'))    # repo root (for my_utils, etc.)
# keep legacy relatives too (harmless if cwd==script dir)
sys.path.append('..')
sys.path.append('../..')

from subpixr.utils import get_general_refiner, standardize_path, load_refiner
from s2dnet_refiner import load_s2dnet, extract_pyramid, refine_hierarchical
# to be added:
# For Local Refiners (Your model, CAPS, S2DNet, LK, NCC)
# CAPS (Coarse-to-fine Attention for Point correspondences - ECCV 2020)
# COTR / ECOTR (Correspondence Transformer - ICCV 2021 / CVPR 2022)
#
#  (Sparse-to-Dense Network - ECCV 2020)
# DeDoDe (Detect, Don't Describe - 3DV 2024)

# For Global Matchers (LightGlue, LoFTR, Patch2Pix)
# LightGlue (ICCV 2023) + SuperPoint
# loftr
# patch2pix

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("--num_iters", type=int, default=1)
parser.add_argument("--perturb_max", type=float, default=16.0,
                    help="Max pixel perturbation applied to the GT location.")
# ~116 sequences × 5 pairs × 10 patches = 5800
parser.add_argument("--num_patches_per_pair", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--refiner_path", type=str, default=None)
parser.add_argument("--hpatches_root", type=str,
                    default="./datasets/hpatches-sequences-release")
parser.add_argument("--skip_baselines", action="store_true", default=False,
                    help="Skip classical baselines (LK, NCC, SIFT, ORB) and load from cache.")
parser.add_argument("--eval_batch_size", type=int, default=100)

parser.add_argument("--skip_patch2pix", action="store_true", default=False,
                    help="Skip patch2pix baseline evaluation to save time.")
parser.add_argument("--skip_caps", action="store_true", default=False,
                    help="Skip CAPS baseline evaluation to save time.")
parser.add_argument("--skip_s2dnet", action="store_true", default=False,
                    help="Skip S2DNet baseline evaluation to save time.")
parser.add_argument("--skip_loftr", action="store_true", default=False,
                    help="Skip LoFTR baseline evaluation to save time.")
parser.add_argument("--skip_cotr", action="store_true", default=False,
                    help="Skip COTR baseline evaluation to save time.")
parser.add_argument("--cotr_zoom_levels", type=int, default=1,
                    help="COTR zoom iterations. >=3 for sub-pixel precision.")
args, _ = parser.parse_known_args()

random.seed(args.seed)
np.random.seed(args.seed)

# --- CONFIGURATION ---
WEIGHTS_PATH = args.refiner_path if args.refiner_path else get_general_refiner()
HPATCHES_ROOT = args.hpatches_root
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_ITERS = args.num_iters
SEARCH_SIZE = 128 if NUM_ITERS == 1 else 192
PERTURB_MAX = args.perturb_max
NUM_PATCHES_PER_PAIR = args.num_patches_per_pair
EVAL_BATCH_SIZE = args.eval_batch_size
RUN_PATCH2PIX = not args.skip_patch2pix
RUN_CAPS = not args.skip_caps
RUN_S2DNET = not args.skip_s2dnet
RUN_LOFTR = not args.skip_loftr
RUN_COTR = not args.skip_cotr
COTR_ZOOM_LVLS = args.cotr_zoom_levels
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

CKPT_DIR = os.path.dirname(WEIGHTS_PATH)

# --- HELPER FUNCTIONS ---
_NORM_MEAN_T = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_NORM_STD_T = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

_S2D_MAX_DIM = 9999  # cap to avoid OOM on large HPatches sequences
_LOFTR_MAX_DIM = 9999  # resize long edge to this to avoid OOM

def _image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    return (t - _NORM_MEAN_T) / _NORM_STD_T


def calculate_mma(errors, thresholds):
    errors = np.array(errors)
    return {t: np.mean(errors < t) * 100 for t in thresholds}


def draw_cross(img, x, y, color, size=6, thickness=2):
    x, y = int(round(x)), int(round(y))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def draw_cross_oblique(img, x, y, color, size=6, thickness=2):
    x, y = int(round(x)), int(round(y))
    cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness)


# --- CLASSICAL BASELINE FUNCTIONS ---
def compute_ncc_offset(template_bgr, search_bgr):
    tg = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(sg, tg, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    th, tw = tg.shape;
    sh, sw = sg.shape
    return max_loc[0] + tw / 2.0 - (sw / 2.0), max_loc[1] + th / 2.0 - (sh / 2.0)


def compute_dense_feature_offset(template_bgr, search_bgr, extractor, matcher, step=2):
    tg = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = tg.shape;
    sh, sw = sg.shape

    kp_t = [cv2.KeyPoint(tw / 2.0, th / 2.0, 15.0)]
    _, des_t = extractor.compute(tg, kp_t)
    if des_t is None: return 0.0, 0.0

    kp_s = [cv2.KeyPoint(float(x), float(y), 15.0) for y in range(0, sh, step) for x in range(0, sw, step)]
    _, des_s = extractor.compute(sg, kp_s)
    if des_s is None: return 0.0, 0.0

    matches = matcher.match(des_t, des_s)
    best_match = min(matches, key=lambda x: x.distance)
    return kp_s[best_match.trainIdx].pt[0] - (sw / 2.0), kp_s[best_match.trainIdx].pt[1] - (sh / 2.0)


def extract_crop_raw(image_np_rgb: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    x1, y1 = int(round(cx - size / 2)), int(round(cy - size / 2))
    h, w = image_np_rgb.shape[:2]
    pad_t, pad_b = max(0, -y1), max(0, (y1 + size) - h)
    pad_l, pad_r = max(0, -x1), max(0, (x1 + size) - w)
    if pad_t or pad_b or pad_l or pad_r:
        image_np_rgb = cv2.copyMakeBorder(image_np_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        x1 += pad_l;
        y1 += pad_t
    patch = image_np_rgb[y1:y1 + size, x1:x1 + size]
    if patch.shape[0] != size or patch.shape[1] != size: patch = cv2.resize(patch, (size, size))
    return cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)


# --- EVAL POINT CACHE ---
def _build_or_load_eval_points(sequences, hpatches_root, seed, perturb_max, num_patches, search_size, script_dir):
    cache_fname = f"eval_points_cache_seed{seed}_perturb{perturb_max:.0f}_n{num_patches}.json"
    cache_path = os.path.join(script_dir, cache_fname)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f: points = json.load(f)
        print(f"[cache] Loaded {len(points)} eval points from {cache_path}")
        return points

    rng = random.Random(seed)
    points = []
    margin = search_size // 2
    for seq in tqdm(sequences, desc="Building eval-point cache"):
        ref_img = cv2.imread(os.path.join(seq, "1.ppm"))
        if ref_img is None: continue
        h_ref, w_ref = ref_img.shape[:2]
        seq_name = os.path.basename(seq)
        for i in range(2, 7):
            trg_img_path, h_path = os.path.join(seq, f"{i}.ppm"), os.path.join(seq, f"H_1_{i}")
            if not os.path.exists(trg_img_path) or not os.path.exists(h_path): continue
            trg_img = cv2.imread(trg_img_path)
            if trg_img is None: continue
            h_trg, w_trg = trg_img.shape[:2]
            H_matrix = np.loadtxt(h_path)
            sampled = attempts = 0
            while sampled < num_patches and attempts < num_patches * 20:
                attempts += 1
                cx1, cy1 = rng.randint(64, w_ref - 64), rng.randint(64, h_ref - 64)
                pt1 = np.array([cx1, cy1, 1.0])
                pt2 = H_matrix @ pt1
                pt2 = pt2 / pt2[2]
                cx2_gt, cy2_gt = float(pt2[0]), float(pt2[1])
                if not (margin < cx2_gt < w_trg - margin and margin < cy2_gt < h_trg - margin): continue
                px, py = rng.uniform(-perturb_max, perturb_max), rng.uniform(-perturb_max, perturb_max)
                points.append(
                    {"seq": seq_name, "pair": i, "cx1": cx1, "cy1": cy1, "cx2_gt": cx2_gt, "cy2_gt": cy2_gt, "px": px,
                     "py": py})
                sampled += 1

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(points, f)
    print(f"[cache] Saved {len(points)} eval points to {cache_path}")
    return points


def main():
    try:
        model, scale_factor = load_refiner(WEIGHTS_PATH, device=DEVICE, force_pmr=False)
    except Exception as e:
        print(f"[SKIP] Cannot load refiner {WEIGHTS_PATH}: {e}", file=sys.stderr)
        sys.exit(2)

    # --- Patch2Pix loading ---
    patch2pix_model = _p2p_estimate_fn = None
    if RUN_PATCH2PIX:
        _P2P_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'patch2pix'))
        _P2P_CKPT = os.path.join(_P2P_ROOT, 'pretrained', 'patch2pix_pretrained.pth')
        try:
            _saved_sys_path = sys.path[:]
            sys.path = [_P2P_ROOT] + [p for p in sys.path if p not in ('', '.', '..')]
            from utils.eval.model_helper import load_model as _p2p_load_fn
            from utils.eval.model_helper import estimate_matches as _p2p_estimate_fn
            sys.path = _saved_sys_path
            if os.path.exists(_P2P_CKPT):
                patch2pix_model = _p2p_load_fn(_P2P_CKPT, method='patch2pix')
                print(f"[patch2pix] Loaded from {_P2P_CKPT}")
        except Exception:
            if '_saved_sys_path' in dir(): sys.path = _saved_sys_path
    else:
        print("[patch2pix] Skipped via local param.")
    _p2p_available = patch2pix_model is not None

    # --- CAPS loading ---
    caps_model = None
    if RUN_CAPS:
        import types
        _CAPS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'caps'))
        _CAPS_CKPT = os.path.join(_CAPS_ROOT, 'weights', 'caps-pretrained.pth')
        try:
            _saved_sys_path = sys.path[:]
            sys.path.insert(0, os.path.join(_CAPS_ROOT, 'CAPS'))
            from network import CAPSNet
            sys.path = _saved_sys_path
            _caps_args = types.SimpleNamespace(
                pretrained=1, backbone='resnet50', coarse_feat_dim=128, fine_feat_dim=128,
                prob_from='correlation', use_nn=1, window_size=0.125,
            )
            caps_model = CAPSNet(_caps_args, DEVICE).eval().to(DEVICE)
            _ckpt = torch.load(_CAPS_CKPT, map_location=DEVICE)
            caps_model.load_state_dict(_ckpt.get("state_dict", _ckpt))
            print(f"[CAPS] Loaded from {_CAPS_CKPT}")
        except Exception as _caps_err:
            if '_saved_sys_path' in dir(): sys.path = _saved_sys_path
            print(f"[CAPS] Skipped: {_caps_err}")
    else:
        print("[CAPS] Skipped via local param.")
    _caps_available = caps_model is not None

    # --- S2DNet loading ---
    # Full hierarchical S2DNet pipeline (coarse→mid→fine with cycle filter),
    # shared with eval_S2DNet_sota.py so MegaDepth/ScanNet and HPatches use
    # the same refinement logic. Pyramid extraction is amortised per image.
    s2dnet_model = None
    s2d_level_scales = None
    if RUN_S2DNET:
        _S2D_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'S2DNet-Minimal'))
        _S2D_CKPT = os.path.join(_S2D_ROOT, 's2dnet_weights.pth')
        try:
            s2dnet_model, s2d_level_scales = load_s2dnet(_S2D_CKPT, DEVICE)
            print(f"[S2DNet] Loaded from {_S2D_CKPT}")
            print(f"[S2DNet] Level strides: {s2d_level_scales}")
        except Exception as _s2d_err:
            print(f"[S2DNet] Skipped: {_s2d_err}")
    else:
        print("[S2DNet] Skipped via local param.")
    _s2dnet_available = s2dnet_model is not None



    def _s2d_extract(img_rgb: np.ndarray) -> tuple:
        """Extract S2DNet pyramid with optional downscale to avoid OOM.

        Returns (pyr, scale) where scale is the factor applied to the image
        (1.0 if no resize was needed). Callers must pass scale into _s2d_refine
        so that keypoint coordinates are correctly mapped into feature space.
        """
        h, w = img_rgb.shape[:2]
        max_dim = max(h, w)
        if max_dim > _S2D_MAX_DIM:
            scale = _S2D_MAX_DIM / max_dim
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
        return extract_pyramid(img_rgb, s2dnet_model, DEVICE), scale

    def _s2d_refine(s2d_ref_pyr: list, s2d_trg_pyr: list,
                    ref_scale: float, trg_scale: float,
                    cx1: float, cy1: float, cx2_init: float, cy2_init: float) -> tuple:
        """Hierarchical coarse-to-fine refinement for a single point.

        Returns (dx, dy) relative to (cx2_init, cy2_init) in original pixel coords.
        Keypoints are scaled into pyramid space before search; the returned delta
        is scaled back to original image space.
        """
        kp0 = np.array([[cx1 * ref_scale, cy1 * ref_scale]], dtype=np.float32)
        kp1 = np.array([[cx2_init * trg_scale, cy2_init * trg_scale]], dtype=np.float32)
        deltas, _cycle_ok = refine_hierarchical(
            s2d_ref_pyr, s2d_trg_pyr, kp0, kp1, s2d_level_scales,
        )
        return float(deltas[0, 0]) / trg_scale, float(deltas[0, 1]) / trg_scale

    # --- LoFTR loading ---
    # LoFTR is semi-dense: runs once per pair, produces matches at ~8px grid → good coverage.
    loftr_matcher = None
    if RUN_LOFTR:
        try:
            import kornia.feature as KF
            loftr_matcher = KF.LoFTR(pretrained='outdoor').eval().to(DEVICE)
            print("[LoFTR] Loaded (outdoor weights)")
        except Exception as _loftr_err:
            print(f"[LoFTR] Skipped: {_loftr_err}")
    else:
        print("[LoFTR] Skipped via local param.")
    _loftr_available = loftr_matcher is not None



    def _loftr_run_pair(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> np.ndarray:
        """Run LoFTR on a full image pair. Returns matches [N,4] in original pixel coords."""
        def _prep(img):
            h, w = img.shape[:2]
            scale = min(1.0, _LOFTR_MAX_DIM / max(h, w))
            nh, nw = int(h * scale) // 8 * 8, int(w * scale) // 8 * 8
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if (nh, nw) != (h, w):
                gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(gray).float()[None, None].to(DEVICE) / 255.0
            return t, w / nw, h / nh  # tensor, scale_x_back, scale_y_back
        t1, sx1, sy1 = _prep(img1_rgb)
        t2, sx2, sy2 = _prep(img2_rgb)
        try:
            with torch.no_grad():
                out = loftr_matcher({'image0': t1, 'image1': t2})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return np.empty((0, 4), dtype=np.float32)
        kp0 = out['keypoints0'].cpu().numpy()  # [N, 2] in resized coords
        kp1 = out['keypoints1'].cpu().numpy()  # [N, 2] in resized coords
        if len(kp0) == 0:
            return np.empty((0, 4), dtype=np.float32)
        kp0 = kp0 * np.array([sx1, sy1])
        kp1 = kp1 * np.array([sx2, sy2])
        return np.concatenate([kp0, kp1], axis=1).astype(np.float32)  # [N, 4]: x1,y1,x2,y2

    # --- COTR loading ---
    # COTR is a query-driven transformer refiner (ICCV 2021). Runs in multiscale
    # mode (zoom=2) for sub-pixel precision. Per-pair batched via SparseEngine.
    cotr_model = cotr_engine = None
    if RUN_COTR:
        torch.backends.cudnn.benchmark = True
        _COTR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'COTR'))
        _COTR_CKPT = os.path.join(_COTR_ROOT, 'out', 'default', 'checkpoint.pth.tar')
        try:
            if not os.path.isdir(_COTR_ROOT):
                raise FileNotFoundError(f"COTR repo not found at: {_COTR_ROOT}")
            if not os.path.isfile(_COTR_CKPT):
                raise FileNotFoundError(f"COTR checkpoint not found: {_COTR_CKPT}")

            # Patch global_configs to avoid hard assertion on missing out dirs
            _gcfg_path = os.path.join(_COTR_ROOT, "COTR", "global_configs", "__init__.py")
            with open(_gcfg_path, "r") as _f:
                _gcfg = _f.read()
            _ASSERT_OUT = "assert os.path.isdir(general_config['out']), f'Please create {general_config[\"out\"]}'"
            _ASSERT_TB = "assert os.path.isdir(general_config['tb_out']), f'Please create {general_config[\"tb_out\"]}'"
            _MAKEDIRS_OUT = "os.makedirs(general_config['out'], exist_ok=True)"
            _MAKEDIRS_TB = "os.makedirs(general_config['tb_out'], exist_ok=True)"
            if _ASSERT_OUT in _gcfg or _ASSERT_TB in _gcfg:
                _gcfg = _gcfg.replace(_ASSERT_OUT, _MAKEDIRS_OUT).replace(_ASSERT_TB, _MAKEDIRS_TB)
                with open(_gcfg_path, "w") as _f:
                    _f.write(_gcfg)

            # Patch capture.py to make pytables optional
            _capture_path = os.path.join(_COTR_ROOT, "COTR", "cameras", "capture.py")
            with open(_capture_path, "r") as _f:
                _capture = _f.read()
            _TABLES_HARD = "import tables"
            _TABLES_SOFT = ("try:\n    import tables\nexcept ImportError:\n    tables = None")
            if _TABLES_HARD in _capture and _TABLES_SOFT not in _capture:
                _capture = _capture.replace(_TABLES_HARD, _TABLES_SOFT)
                with open(_capture_path, "w") as _f:
                    _f.write(_capture)

            # Patch deprecated np.int / np.bool -> builtins in sparse_engine.py (numpy >= 1.20)
            _sparse_path = os.path.join(_COTR_ROOT, "COTR", "inference", "sparse_engine.py")
            with open(_sparse_path, "r") as _f:
                _sparse = _f.read()
            _orig_sparse = _sparse
            _sparse = _sparse.replace("dtype=np.int)", "dtype=int)")
            _sparse = _sparse.replace(".astype(np.bool)", ".astype(bool)")
            if _sparse != _orig_sparse:
                with open(_sparse_path, "w") as _f:
                    _f.write(_sparse)

            _saved_sys_path = sys.path[:]
            sys.path.insert(0, _COTR_ROOT)
            from COTR.models import build_model
            from COTR.options.options import set_COTR_arguments
            from COTR.inference.sparse_engine import SparseEngine
            from COTR.inference.refinement_task import RefinementTask
            from COTR.utils.utils import safe_load_weights
            sys.path = _saved_sys_path

            _cotr_parser = argparse.ArgumentParser()
            set_COTR_arguments(_cotr_parser)
            _cotr_opt, _ = _cotr_parser.parse_known_args([])
            _layer_channels = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
            _cotr_opt.dim_feedforward = _layer_channels[_cotr_opt.layer]

            cotr_model = build_model(_cotr_opt).to(DEVICE).eval()
            _weights = torch.load(_COTR_CKPT, map_location=DEVICE, weights_only=False)['model_state_dict']
            safe_load_weights(cotr_model, _weights)
            cotr_engine = SparseEngine(cotr_model, batch_size=256, mode='tile')
            print(f"[COTR] Loaded from {_COTR_CKPT}  (zoom_levels={COTR_ZOOM_LVLS})")
        except Exception as _cotr_err:
            if '_saved_sys_path' in dir(): sys.path = _saved_sys_path
            print(f"[COTR] Skipped: {_cotr_err}")
            cotr_model = cotr_engine = None
    else:
        print("[COTR] Skipped via local param.")
    _cotr_available = cotr_model is not None and cotr_engine is not None

    def _crop_patch(img: np.ndarray, cx: float, cy: float, half: int):
        """Crop a (2*half)×(2*half) window around (cx,cy), clamped to image bounds.
        Returns (crop, x0, y0) where (x0,y0) is the top-left corner used."""
        H, W = img.shape[:2]
        x0 = int(round(cx)) - half
        y0 = int(round(cy)) - half
        x0 = max(0, min(W - 2 * half, x0))
        y0 = max(0, min(H - 2 * half, y0))
        return img[y0:y0 + 2 * half, x0:x0 + 2 * half], x0, y0

    def _cotr_refine_pair(img0_rgb: np.ndarray, img1_rgb: np.ndarray,
                          kp0_px: np.ndarray, kp1_init_px: np.ndarray) -> np.ndarray:
        """Multiscale COTR refinement constrained to SEARCH_SIZE window per keypoint.
        Each keypoint gets its own crop (same search radius as NCC/LK/S2DNet).
        Returns refined kp1 positions in original-image pixel coords [N,2]."""
        N = len(kp0_px)
        if N == 0:
            return kp1_init_px.copy()

        half = SEARCH_SIZE // 2  # same window as NCC / LK / S2DNet
        _COTR_MAX_SZ = 256
        zoom_ins = np.linspace(0.5, 0.0625, COTR_ZOOM_LVLS)
        refined = kp1_init_px.copy().astype(np.float64)

        tasks = []
        offsets = []  # (x0_ref, y0_ref, x0_trg, y0_trg) per task
        for i in range(N):
            crop0, x0_r, y0_r = _crop_patch(img0_rgb, kp0_px[i, 0], kp0_px[i, 1], half)
            crop1, x0_t, y0_t = _crop_patch(img1_rgb, kp1_init_px[i, 0], kp1_init_px[i, 1], half)
            h0c, w0c = crop0.shape[:2]
            h1c, w1c = crop1.shape[:2]
            area_0 = (_COTR_MAX_SZ ** 2) / float(h0c * w0c)
            area_1 = (_COTR_MAX_SZ ** 2) / float(h1c * w1c)
            q0 = np.array([kp0_px[i, 0] - x0_r, kp0_px[i, 1] - y0_r], dtype=np.float64)
            q1 = np.array([kp1_init_px[i, 0] - x0_t, kp1_init_px[i, 1] - y0_t], dtype=np.float64)
            tasks.append(RefinementTask(crop0, crop1, q0, q1,
                                        area_from=area_0, area_to=area_1,
                                        converge_iters=1, zoom_ins=zoom_ins, identifier=i))
            offsets.append((x0_t, y0_t))

        try:
            while True:
                with contextlib.redirect_stdout(io.StringIO()):
                    task_ref, img_batch, query_batch = cotr_engine.form_batch(tasks)
                if len(task_ref) == 0:
                    break
                with torch.no_grad():
                    out = cotr_engine.infer_batch(img_batch, query_batch)
                for t, o in zip(task_ref, out):
                    t.step(o)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return kp1_init_px.copy().astype(np.float32)

        for t in tasks:
            result = t.conclude(force=True)
            if result is not None:
                ox, oy = offsets[t.identifier]
                refined[t.identifier] = [result[2] + ox, result[3] + oy]
        return refined.astype(np.float32)

    def _caps_refine(img1_rgb: np.ndarray, img2_rgb: np.ndarray, cx1: float, cy1: float,
                     cx2_init: float, cy2_init: float):
        """Run CAPS on full image pair; returns (dx, dy) relative to (cx2_init, cy2_init)."""
        h1, w1 = img1_rgb.shape[:2]
        h2, w2 = img2_rgb.shape[:2]

        # 1. Apply proper ImageNet normalization using your existing helper
        t1 = _image_to_tensor(img1_rgb)
        t2 = _image_to_tensor(img2_rgb)

        # 2. CAPS .test() takes pixel-space query coords and returns pixel-space
        #    correspondences (it does the [-1,1] normalize/denormalize internally,
        #    see network.py: normalize(coord1, h1i, w1i) ... denormalize(coord2_ef_n, h2i, w2i)).
        kp_q = torch.tensor([[[cx1, cy1]]], dtype=torch.float32, device=DEVICE)

        # 3. Inference
        coord2_ef, _ = caps_model.test(t1, t2, kp_q)

        # 4. Output is already in pixel coordinates of img2.
        x2 = float(coord2_ef[0, 0, 0].item())
        y2 = float(coord2_ef[0, 0, 1].item())

        # 5. Return the delta relative to the initial guess
        return x2 - cx2_init, y2 - cy2_init
    # patch2pix: run once per image pair, then do fast per-point lookup.
    # imsize=1024 matches the official Patch2Pix HPatches eval (imsize is the
    # longer-edge target; matches come back in original-image coords). Same
    # value used in Strategy A for direct comparability.
    # On OOM, the pair is skipped and all its points get (0,0) — no artificial favouring.
    _P2P_IMSIZE = 1024
    _p2p_oom_pairs = 0

    def _p2p_run_pair(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> np.ndarray:
        """Run patch2pix on a full image pair. Returns matches [N,4] in original pixel coords,
        or empty array on CUDA OOM (counted separately, not silently hidden)."""
        nonlocal _p2p_oom_pairs
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as _f1, \
             tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as _f2:
            cv2.imwrite(_f1.name, cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(_f2.name, cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
            _tmp1, _tmp2 = _f1.name, _f2.name
        try:
            _m, _, _ = _p2p_estimate_fn(patch2pix_model, _tmp1, _tmp2, ksize=2, io_thres=0.25,
                                        eval_type='fine', imsize=_P2P_IMSIZE)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _p2p_oom_pairs += 1
            _m = np.empty((0, 4), dtype=np.float32)
        finally:
            os.unlink(_tmp1)
            os.unlink(_tmp2)
        return _m  # [N, 4]: (x1, y1, x2, y2) in original pixel coords

    def _p2p_lookup(pair_matches: np.ndarray, cx1: float, cy1: float,
                    cx2_init: float, cy2_init: float, max_query_dist: float = 32.0):
        """Find nearest match to (cx1,cy1) in img1; return (dx,dy) relative to cx2_init,cy2_init.
        Falls back to (0,0) if no match is within max_query_dist px — patch2pix is sparse so a
        distant 'nearest' match is unreliable garbage, not a refinement."""
        if len(pair_matches) == 0:
            return 0.0, 0.0
        _dists = np.linalg.norm(pair_matches[:, :2] - np.array([cx1, cy1]), axis=1)
        _best = int(np.argmin(_dists))
        if _dists[_best] > max_query_dist:
            return 0.0, 0.0  # no confident match near this query point
        return float(pair_matches[_best, 2]) - cx2_init, float(pair_matches[_best, 3]) - cy2_init

    sequences = sorted(glob.glob(os.path.join(HPATCHES_ROOT, "*")))
    if not sequences: raise FileNotFoundError("No sequences found.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_points = _build_or_load_eval_points(sequences, HPATCHES_ROOT, args.seed, PERTURB_MAX, NUM_PATCHES_PER_PAIR,
                                             SEARCH_SIZE, script_dir)

    _baselines_cache_path = os.path.join(script_dir,
                                         f"baselines_cache_seed{args.seed}_perturb{PERTURB_MAX:.0f}_n{NUM_PATCHES_PER_PAIR}.json")
    _baselines_cache = None
    if args.skip_baselines and os.path.exists(_baselines_cache_path):
        with open(_baselines_cache_path, "r", encoding="utf-8") as _f: _baselines_cache = json.load(_f)
        print(f"[cache] Loaded baselines for {len(_baselines_cache)} points")
    _baselines_computed = []

    _p2p_cache_path = os.path.join(script_dir,
                                   f"p2p_mma_cache_seed{args.seed}_perturb{PERTURB_MAX:.0f}_n{NUM_PATCHES_PER_PAIR}.json")
    _p2p_cache = _p2p_cache_computed = None
    if os.path.exists(_p2p_cache_path):
        with open(_p2p_cache_path, "r", encoding="utf-8") as _pf:
            _p2p_cache = json.load(_pf)
        print(f"[patch2pix cache] Loaded {len(_p2p_cache)} entries")
    elif _p2p_available:
        _p2p_cache_computed = []
        print(f"[patch2pix cache] Will build on this run.")

    _caps_cache_path = os.path.join(script_dir,
                                    f"caps_mma_cache_seed{args.seed}_perturb{PERTURB_MAX:.0f}_n{NUM_PATCHES_PER_PAIR}.json")
    _caps_cache = _caps_cache_computed = None
    if os.path.exists(_caps_cache_path):
        with open(_caps_cache_path, "r", encoding="utf-8") as _cf:
            _caps_cache = json.load(_cf)
        print(f"[CAPS cache] Loaded {len(_caps_cache)} entries")
    elif _caps_available:
        _caps_cache_computed = []
        print(f"[CAPS cache] Will build on this run.")

    # S2DNet has no separate cache — features are stored per-image (not per-point),
    # so they are recomputed on each run. Per-point results are cached like the others.
    _s2d_cache_path = os.path.join(script_dir,
                                   f"s2d_mma_cache_hier_d{_S2D_MAX_DIM}_seed{args.seed}_perturb{PERTURB_MAX:.0f}_n{NUM_PATCHES_PER_PAIR}.json")
    _s2d_cache = _s2d_cache_computed = None
    if os.path.exists(_s2d_cache_path):
        with open(_s2d_cache_path, "r", encoding="utf-8") as _sf:
            _s2d_cache = json.load(_sf)
        print(f"[S2DNet cache] Loaded {len(_s2d_cache)} entries")
    elif _s2dnet_available:
        _s2d_cache_computed = []
        print(f"[S2DNet cache] Will build on this run.")

    _loftr_cache_path = os.path.join(script_dir,
                                     f"loftr_mma_cache_d{_LOFTR_MAX_DIM}_seed{args.seed}_perturb{PERTURB_MAX:.0f}_n{NUM_PATCHES_PER_PAIR}.json")
    _loftr_cache = _loftr_cache_computed = None
    if os.path.exists(_loftr_cache_path):
        with open(_loftr_cache_path, "r", encoding="utf-8") as _lf:
            _loftr_cache = json.load(_lf)
        print(f"[LoFTR cache] Loaded {len(_loftr_cache)} entries")
    elif _loftr_available:
        _loftr_cache_computed = []
        print(f"[LoFTR cache] Will build on this run.")

    _cotr_cache_path = os.path.join(script_dir,
                                    f"cotr_mma_cache_z{COTR_ZOOM_LVLS}_seed{args.seed}_perturb{PERTURB_MAX:.0f}_n{NUM_PATCHES_PER_PAIR}.json")
    _cotr_cache = _cotr_cache_computed = None
    if os.path.exists(_cotr_cache_path):
        with open(_cotr_cache_path, "r", encoding="utf-8") as _cf:
            _cotr_cache = json.load(_cf)
        print(f"[COTR cache] Loaded {len(_cotr_cache)} entries")
    elif _cotr_available:
        _cotr_cache_computed = []
        print(f"[COTR cache] Will build on this run.")

    # Pre-group eval points by (seq, pair) so COTR can batch-process per pair.
    from collections import defaultdict
    _pair_to_indices = defaultdict(list)
    for _i, _p in enumerate(eval_points):
        _pair_to_indices[(_p["seq"], _p["pair"])].append(_i)

    err_initial, err_lk, err_ncc, err_sift, err_orb, err_p2p, err_caps, err_s2d, err_loftr, err_cotr, err_net = [], [], [], [], [], [], [], [], [], [], []

    _cur_seq_name = _cur_pair = None
    ref_img_rgb = trg_img_rgb = ref_tensor = trg_tensor = ref_gray_lk = trg_gray_lk = None
    s2d_ref_pyr = s2d_trg_pyr = None  # hyperfeature pyramids, recomputed per image
    s2d_ref_scale = s2d_trg_scale = 1.0  # resize scales used during extraction
    _p2p_pair_matches = np.empty((0, 4), dtype=np.float32)  # matches for current pair, recomputed per pair
    _loftr_pair_matches = np.empty((0, 4), dtype=np.float32)  # matches for current pair, recomputed per pair
    _using_cached_baselines = _baselines_cache is not None
    _using_cached_p2p = _p2p_cache is not None
    _using_cached_caps = _caps_cache is not None
    _using_cached_s2d = _s2d_cache is not None
    _using_cached_loftr = _loftr_cache is not None
    _using_cached_cotr = _cotr_cache is not None
    _cotr_pair_results = {}  # {pt_idx: (refined_x, refined_y)}

    sift_extractor = sift_matcher = orb_extractor = orb_matcher = None
    if not _using_cached_baselines:
        sift_extractor = cv2.SIFT_create()
        sift_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        orb_extractor = cv2.ORB_create(edgeThreshold=0, patchSize=15)
        orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    _pending = []

    def _flush_batch():
        if not _pending: return
        B = len(_pending)

        pts_t = torch.tensor([[p["cx1"], p["cy1"]] for p in _pending], device=DEVICE)
        pts_s = torch.tensor([[p["cx2_init"], p["cy2_init"]] for p in _pending], device=DEVICE)

        tl_t = torch.round(pts_t) - 64.0
        tl_s = torch.round(pts_s) - (SEARCH_SIZE / 2.0)

        b_ref = ref_tensor.expand(B, -1, -1, -1)
        b_trg = trg_tensor.expand(B, -1, -1, -1)

        templates = model.extract_patch(b_ref, tl_t, 128)
        searches = model.extract_patch(b_trg, tl_s, SEARCH_SIZE)

        batch_preds = model(templates, searches, num_iters=NUM_ITERS)
        batch_deltas = batch_preds.detach().cpu().numpy() * scale_factor

        for p, delta in zip(_pending, batch_deltas):
            refined_x = round(p["cx2_init"]) + delta[0]
            refined_y = round(p["cy2_init"]) + delta[1]
            epe_net = np.linalg.norm([refined_x - p["cx2_gt"], refined_y - p["cy2_gt"]])
            err_net.append(epe_net)

        _pending.clear()

    with torch.no_grad():
        for _pt_idx, pt in enumerate(tqdm(eval_points, desc="Evaluating Strategy B")):
            seq_name = pt["seq"]
            pair_i = pt["pair"]

            if seq_name != _cur_seq_name or pair_i != _cur_pair:
                _flush_batch()

                if seq_name != _cur_seq_name:
                    ref_img = cv2.imread(os.path.join(HPATCHES_ROOT, seq_name, "1.ppm"))
                    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    ref_tensor = _image_to_tensor(ref_img_rgb)
                    if not _using_cached_baselines: ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    if _s2dnet_available and not _using_cached_s2d:
                        s2d_ref_pyr, s2d_ref_scale = _s2d_extract(ref_img_rgb)
                    _cur_seq_name = seq_name
                    _cur_pair = None

                if pair_i != _cur_pair:
                    trg_img = cv2.imread(os.path.join(HPATCHES_ROOT, seq_name, f"{pair_i}.ppm"))
                    trg_img_rgb = cv2.cvtColor(trg_img, cv2.COLOR_BGR2RGB)
                    trg_tensor = _image_to_tensor(trg_img_rgb)
                    if not _using_cached_baselines:
                        trg_img_gray = cv2.cvtColor(trg_img, cv2.COLOR_BGR2GRAY)
                        h_ref, w_ref = ref_img_gray.shape
                        h_trg, w_trg = trg_img_gray.shape
                        max_h, max_w = max(h_ref, h_trg), max(w_ref, w_trg)
                        ref_gray_lk = cv2.copyMakeBorder(ref_img_gray, 0, max_h - h_ref, 0, max_w - w_ref,
                                                         cv2.BORDER_REPLICATE)
                        trg_gray_lk = cv2.copyMakeBorder(trg_img_gray, 0, max_h - h_trg, 0, max_w - w_trg,
                                                         cv2.BORDER_REPLICATE)
                    # Run patch2pix once per pair (not per point) to avoid redundant inference
                    if _p2p_available and not _using_cached_p2p:
                        _p2p_pair_matches = _p2p_run_pair(ref_img_rgb, trg_img_rgb)
                    # Run LoFTR once per pair
                    if _loftr_available and not _using_cached_loftr:
                        _loftr_pair_matches = _loftr_run_pair(ref_img_rgb, trg_img_rgb)
                    # Run COTR once per pair on all queries belonging to this pair
                    if _cotr_available and not _using_cached_cotr:
                        _pair_idxs = _pair_to_indices[(seq_name, pair_i)]
                        _kp0_arr = np.array(
                            [[eval_points[_i]["cx1"], eval_points[_i]["cy1"]] for _i in _pair_idxs],
                            dtype=np.float32)
                        _kp1_init_arr = np.array(
                            [[eval_points[_i]["cx2_gt"] + eval_points[_i]["px"],
                              eval_points[_i]["cy2_gt"] + eval_points[_i]["py"]] for _i in _pair_idxs],
                            dtype=np.float32)
                        _refined_arr = _cotr_refine_pair(ref_img_rgb, trg_img_rgb,
                                                         _kp0_arr, _kp1_init_arr)
                        _cotr_pair_results = {
                            _pair_idxs[_k]: (float(_refined_arr[_k, 0]), float(_refined_arr[_k, 1]))
                            for _k in range(len(_pair_idxs))
                        }
                    # Extract S2DNet target features once per pair
                    if _s2dnet_available and not _using_cached_s2d:
                        s2d_trg_pyr, s2d_trg_scale = _s2d_extract(trg_img_rgb)
                    _cur_pair = pair_i

            cx1, cy1 = pt["cx1"], pt["cy1"]
            cx2_gt, cy2_gt = pt["cx2_gt"], pt["cy2_gt"]
            cx2_init, cy2_init = cx2_gt + pt["px"], cy2_gt + pt["py"]

            err_initial.append(np.linalg.norm([pt["px"], pt["py"]]))
            snap_x_f, snap_y_f = float(round(cx2_init)), float(round(cy2_init))

            template_raw = search_raw = None
            if not _using_cached_baselines:
                template_raw = extract_crop_raw(ref_img_rgb, cx1, cy1, 128)
                search_raw = extract_crop_raw(trg_img_rgb, cx2_init, cy2_init, SEARCH_SIZE)

            # Baseline classical processing
            if _using_cached_baselines and _pt_idx < len(_baselines_cache):
                _bc = _baselines_cache[_pt_idx]
                lk_dx_abs, lk_dy_abs = _bc["lk_dx_abs"], _bc["lk_dy_abs"]
                ncc_dx, ncc_dy = _bc["ncc_dx"], _bc["ncc_dy"]
                sift_dx, sift_dy = _bc["sift_dx"], _bc["sift_dy"]
                orb_dx, orb_dy = _bc["orb_dx"], _bc["orb_dy"]

                err_lk.append(np.linalg.norm([lk_dx_abs - cx2_gt, lk_dy_abs - cy2_gt]))
                err_ncc.append(np.linalg.norm([cx2_init + ncc_dx - cx2_gt, cy2_init + ncc_dy - cy2_gt]))
                err_sift.append(np.linalg.norm([cx2_init + sift_dx - cx2_gt, cy2_init + sift_dy - cy2_gt]))
                err_orb.append(np.linalg.norm([cx2_init + orb_dx - cx2_gt, cy2_init + orb_dy - cy2_gt]))
            else:
                p0 = np.array([[[cx1, cy1]]], dtype=np.float32)
                p1_init = np.array([[[cx2_init, cy2_init]]], dtype=np.float32)
                p1_lk, st, _ = cv2.calcOpticalFlowPyrLK(ref_gray_lk, trg_gray_lk, p0, p1_init, winSize=(21, 21),
                                                        maxLevel=3,
                                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30,
                                                                  0.01), flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                lk_pt = p1_lk[0][0] if st[0][0] == 1 else np.array([cx2_init, cy2_init])
                err_lk.append(np.linalg.norm(lk_pt - np.array([cx2_gt, cy2_gt])))
                lk_dx_abs, lk_dy_abs = float(lk_pt[0]), float(lk_pt[1])

                ncc_dx, ncc_dy = compute_ncc_offset(template_raw, search_raw)
                err_ncc.append(np.linalg.norm([cx2_init + ncc_dx - cx2_gt, cy2_init + ncc_dy - cy2_gt]))

                sift_dx, sift_dy = compute_dense_feature_offset(template_raw, search_raw, sift_extractor, sift_matcher,
                                                                step=2)
                err_sift.append(np.linalg.norm([cx2_init + sift_dx - cx2_gt, cy2_init + sift_dy - cy2_gt]))

                orb_dx, orb_dy = compute_dense_feature_offset(template_raw, search_raw, orb_extractor, orb_matcher,
                                                              step=2)
                err_orb.append(np.linalg.norm([cx2_init + orb_dx - cx2_gt, cy2_init + orb_dy - cy2_gt]))

                _baselines_computed.append({
                    "lk_dx_abs": lk_dx_abs, "lk_dy_abs": lk_dy_abs,
                    "ncc_dx": ncc_dx, "ncc_dy": ncc_dy,
                    "sift_dx": sift_dx, "sift_dy": sift_dy,
                    "orb_dx": orb_dx, "orb_dy": orb_dy,
                })

            # Patch2Pix processing
            if _using_cached_p2p and _pt_idx < len(_p2p_cache):
                p2p_dx, p2p_dy = float(_p2p_cache[_pt_idx]["p2p_dx"]), float(_p2p_cache[_pt_idx]["p2p_dy"])
                err_p2p.append(np.linalg.norm([cx2_init + p2p_dx - cx2_gt, cy2_init + p2p_dy - cy2_gt]))
            elif _p2p_available:
                p2p_dx, p2p_dy = _p2p_lookup(_p2p_pair_matches, cx1, cy1, cx2_init, cy2_init)
                _p2p_cache_computed.append({"p2p_dx": p2p_dx, "p2p_dy": p2p_dy})
                err_p2p.append(np.linalg.norm([cx2_init + p2p_dx - cx2_gt, cy2_init + p2p_dy - cy2_gt]))
            else:
                p2p_dx = p2p_dy = 0.0

            # CAPS processing
            if _using_cached_caps and _pt_idx < len(_caps_cache):
                caps_dx, caps_dy = float(_caps_cache[_pt_idx]["caps_dx"]), float(_caps_cache[_pt_idx]["caps_dy"])
                err_caps.append(np.linalg.norm([cx2_init + caps_dx - cx2_gt, cy2_init + caps_dy - cy2_gt]))
            elif _caps_available:
                caps_dx, caps_dy = _caps_refine(ref_img_rgb, trg_img_rgb, cx1, cy1, cx2_init, cy2_init)
                _caps_cache_computed.append({"caps_dx": caps_dx, "caps_dy": caps_dy})
                err_caps.append(np.linalg.norm([cx2_init + caps_dx - cx2_gt, cy2_init + caps_dy - cy2_gt]))
            else:
                caps_dx = caps_dy = 0.0

            # S2DNet processing
            if _using_cached_s2d and _pt_idx < len(_s2d_cache):
                s2d_dx, s2d_dy = float(_s2d_cache[_pt_idx]["s2d_dx"]), float(_s2d_cache[_pt_idx]["s2d_dy"])
                err_s2d.append(np.linalg.norm([cx2_init + s2d_dx - cx2_gt, cy2_init + s2d_dy - cy2_gt]))
            elif _s2dnet_available:
                s2d_dx, s2d_dy = _s2d_refine(s2d_ref_pyr, s2d_trg_pyr, s2d_ref_scale, s2d_trg_scale, cx1, cy1, cx2_init, cy2_init)
                _s2d_cache_computed.append({"s2d_dx": s2d_dx, "s2d_dy": s2d_dy})
                err_s2d.append(np.linalg.norm([cx2_init + s2d_dx - cx2_gt, cy2_init + s2d_dy - cy2_gt]))
            else:
                s2d_dx = s2d_dy = 0.0

            # LoFTR processing (semi-dense, per-pair lookup — same pattern as patch2pix)
            if _using_cached_loftr and _pt_idx < len(_loftr_cache):
                loftr_dx, loftr_dy = float(_loftr_cache[_pt_idx]["loftr_dx"]), float(_loftr_cache[_pt_idx]["loftr_dy"])
                err_loftr.append(np.linalg.norm([cx2_init + loftr_dx - cx2_gt, cy2_init + loftr_dy - cy2_gt]))
            elif _loftr_available:
                loftr_dx, loftr_dy = _p2p_lookup(_loftr_pair_matches, cx1, cy1, cx2_init, cy2_init)
                _loftr_cache_computed.append({"loftr_dx": loftr_dx, "loftr_dy": loftr_dy})
                err_loftr.append(np.linalg.norm([cx2_init + loftr_dx - cx2_gt, cy2_init + loftr_dy - cy2_gt]))
            else:
                loftr_dx = loftr_dy = 0.0

            # COTR processing (per-pair batched, per-point lookup)
            if _using_cached_cotr and _pt_idx < len(_cotr_cache):
                cotr_dx, cotr_dy = float(_cotr_cache[_pt_idx]["cotr_dx"]), float(_cotr_cache[_pt_idx]["cotr_dy"])
                err_cotr.append(np.linalg.norm([cx2_init + cotr_dx - cx2_gt, cy2_init + cotr_dy - cy2_gt]))
            elif _cotr_available and _pt_idx in _cotr_pair_results:
                _rx, _ry = _cotr_pair_results[_pt_idx]
                cotr_dx, cotr_dy = _rx - cx2_init, _ry - cy2_init
                _cotr_cache_computed.append({"cotr_dx": cotr_dx, "cotr_dy": cotr_dy})
                err_cotr.append(np.linalg.norm([cx2_init + cotr_dx - cx2_gt, cy2_init + cotr_dy - cy2_gt]))
            else:
                cotr_dx = cotr_dy = 0.0

            # Queue for Neural Refiner
            _pending.append({
                "cx1": cx1, "cy1": cy1,
                "cx2_init": cx2_init, "cy2_init": cy2_init,
                "cx2_gt": cx2_gt, "cy2_gt": cy2_gt,
                "snap_x_f": snap_x_f, "snap_y_f": snap_y_f,
                "t_raw": template_raw, "s_raw": search_raw,
                "gt_dx": cx2_gt - snap_x_f, "gt_dy": cy2_gt - snap_y_f,
                "lk_dx_abs": lk_dx_abs, "lk_dy_abs": lk_dy_abs,
                "ncc_dx": ncc_dx, "ncc_dy": ncc_dy,
                "sift_dx": sift_dx, "sift_dy": sift_dy,
                "orb_dx": orb_dx, "orb_dy": orb_dy,
                "p2p_dx": p2p_dx, "p2p_dy": p2p_dy,
                "caps_dx": caps_dx, "caps_dy": caps_dy,
                "s2d_dx": s2d_dx, "s2d_dy": s2d_dy,
                "loftr_dx": loftr_dx, "loftr_dy": loftr_dy,
                "cotr_dx": cotr_dx, "cotr_dy": cotr_dy,
                "perturb_x": pt["px"], "perturb_y": pt["py"],
                "seq_name": seq_name,
            })
            if len(_pending) >= EVAL_BATCH_SIZE:
                _flush_batch()

    # Flush final points
    _flush_batch()

    # Save caches
    if _p2p_available and _p2p_cache_computed and not _using_cached_p2p:
        with open(_p2p_cache_path, "w", encoding="utf-8") as _pf: json.dump(_p2p_cache_computed, _pf)
        print(f"[patch2pix cache] Saved {len(_p2p_cache_computed)} entries")

    if _caps_available and _caps_cache_computed and not _using_cached_caps:
        with open(_caps_cache_path, "w", encoding="utf-8") as _cf: json.dump(_caps_cache_computed, _cf)
        print(f"[CAPS cache] Saved {len(_caps_cache_computed)} entries")

    if _s2dnet_available and _s2d_cache_computed and not _using_cached_s2d:
        with open(_s2d_cache_path, "w", encoding="utf-8") as _sf: json.dump(_s2d_cache_computed, _sf)
        print(f"[S2DNet cache] Saved {len(_s2d_cache_computed)} entries")

    if _loftr_available and _loftr_cache_computed and not _using_cached_loftr:
        with open(_loftr_cache_path, "w", encoding="utf-8") as _lf: json.dump(_loftr_cache_computed, _lf)
        print(f"[LoFTR cache] Saved {len(_loftr_cache_computed)} entries")

    if _cotr_available and _cotr_cache_computed and not _using_cached_cotr:
        with open(_cotr_cache_path, "w", encoding="utf-8") as _cf: json.dump(_cotr_cache_computed, _cf)
        print(f"[COTR cache] Saved {len(_cotr_cache_computed)} entries")

    if _baselines_computed and not _using_cached_baselines:
        with open(_baselines_cache_path, "w", encoding="utf-8") as _f: json.dump(_baselines_computed, _f)
        print(f"[cache] Saved baselines for {len(_baselines_computed)} points")

    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # --- Print Tables ---
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]
    mma_init = calculate_mma(err_initial, thresholds)
    mma_lk = calculate_mma(err_lk, thresholds)
    mma_ncc = calculate_mma(err_ncc, thresholds)
    mma_sift = calculate_mma(err_sift, thresholds)
    mma_orb = calculate_mma(err_orb, thresholds)
    mma_p2p = calculate_mma(err_p2p, thresholds) if err_p2p else None
    mma_caps = calculate_mma(err_caps, thresholds) if err_caps else None
    mma_s2d = calculate_mma(err_s2d, thresholds) if err_s2d else None
    mma_loftr = calculate_mma(err_loftr, thresholds) if err_loftr else None
    mma_cotr = calculate_mma(err_cotr, thresholds) if err_cotr else None
    mma_net = calculate_mma(err_net, thresholds)

    log_name = f"hpatches_strategyB_perturb{PERTURB_MAX:.0f}px.txt"
    log_path = os.path.join(CKPT_DIR, log_name)

    lines = [
        "\n" + "*" * 75,
        f" HPatches Local Refinement Evaluation (Strategy B)",
        f" Initial Perturbation: ±{PERTURB_MAX}px  |  Valid Patches: {len(err_net)}",
        f" Run timestamp : {timestamp}",
        f" Refiner       : {WEIGHTS_PATH}",
        "*" * 75,
        "\n--- Mean End-Point Error (EPE) ---",
        f"Initial (Coarse) : {np.mean(err_initial):.3f} px",
        f"Lucas-Kanade     : {np.mean(err_lk):.3f} px",
        f"NCC Matching     : {np.mean(err_ncc):.3f} px",
        f"Dense SIFT       : {np.mean(err_sift):.3f} px",
        f"Dense ORB        : {np.mean(err_orb):.3f} px",
    ]
    if err_p2p:
        _p2p_oom_note = f"  ({_p2p_oom_pairs} pairs OOM→skipped)" if _p2p_oom_pairs else ""
        lines.append(f"Patch2Pix        : {np.mean(err_p2p):.3f} px{_p2p_oom_note}")
    if err_caps: lines.append(f"CAPS             : {np.mean(err_caps):.3f} px")
    if err_s2d: lines.append(f"S2DNet           : {np.mean(err_s2d):.3f} px")
    if err_loftr: lines.append(f"LoFTR            : {np.mean(err_loftr):.3f} px")
    if err_cotr: lines.append(f"COTR (z{COTR_ZOOM_LVLS})         : {np.mean(err_cotr):.3f} px")
    lines.append(f"Your Refiner     : {np.mean(err_net):.3f} px")
    lines.append("\n--- Mean Matching Accuracy (MMA / PCK) ---")
    lines.append(f"{'Method':<18} | {'0.5px':<6} | {'1px':<6} | {'2px':<6} | {'3px':<6} | {'5px':<6}")
    lines.append("-" * 75)
    lines.append(f"{'Initial (Coarse)':<18} | " + " | ".join([f"{mma_init[t]:.1f}%" for t in thresholds]))
    lines.append(f"{'Lucas-Kanade':<18} | " + " | ".join([f"{mma_lk[t]:.1f}%" for t in thresholds]))
    lines.append(f"{'NCC Matching':<18} | " + " | ".join([f"{mma_ncc[t]:.1f}%" for t in thresholds]))
    lines.append(f"{'Dense SIFT':<18} | " + " | ".join([f"{mma_sift[t]:.1f}%" for t in thresholds]))
    lines.append(f"{'Dense ORB':<18} | " + " | ".join([f"{mma_orb[t]:.1f}%" for t in thresholds]))
    if mma_p2p: lines.append(f"{'Patch2Pix':<18} | " + " | ".join([f"{mma_p2p[t]:.1f}%" for t in thresholds]))
    if mma_caps: lines.append(f"{'CAPS':<18} | " + " | ".join([f"{mma_caps[t]:.1f}%" for t in thresholds]))
    if mma_s2d: lines.append(f"{'S2DNet':<18} | " + " | ".join([f"{mma_s2d[t]:.1f}%" for t in thresholds]))
    if mma_loftr: lines.append(f"{'LoFTR':<18} | " + " | ".join([f"{mma_loftr[t]:.1f}%" for t in thresholds]))
    if mma_cotr: lines.append(f"{f'COTR (z{COTR_ZOOM_LVLS})':<18} | " + " | ".join([f"{mma_cotr[t]:.1f}%" for t in thresholds]))
    lines.append(f"{'Your Refiner':<18} | " + " | ".join([f"{mma_net[t]:.1f}%" for t in thresholds]))
    lines.append("*" * 75)
    lines.append(f"\nLog saved to: {log_path}")

    output = "\n".join(lines)
    print(output)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")


if __name__ == "__main__":
    main()