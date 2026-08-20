"""COTR (Correspondence Transformer, ICCV 2021) wrapper used as a refinement
baseline. Operates on per-pair predictions_baseline.h5 files for MegaDepth or
ScanNet. Forces COTR into a local 256x256 patch mode so it acts as a local
refiner rather than running on the full image, and auto-detects the matcher's
keypoint frame (640 / 1296 / 1600 px) to keep coordinates in 1:1 correspondence.

Setup (one-time):
  git clone https://github.com/ubc-vision/COTR.git
  # Download cotr_default.zip from https://github.com/ubc-vision/COTR/releases
  # Extract so that COTR/out/default/checkpoint.pth.tar exists.
"""
import os
import sys
import argparse
import h5py
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Enable optimized convolutions for fixed-size 256x256 crops
torch.backends.cudnn.benchmark = True

sys.path.append('.')
sys.path.append('..')

from subpixr.utils import standardize_path

# ── Constants ────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
COTR_MAX_SIZE   = 256       # matches COTR's constants.MAX_SIZE
_IMAGENET_MEAN  = [0.485, 0.456, 0.406]
_IMAGENET_STD   = [0.229, 0.224, 0.225]

# ── Arguments ────────────────────────────────────────────────────────────────
_MATCHERS = [
    "superpoint+lightglue-official",
    "superpoint+superglue-official",
    "superpoint+NN",
    "disk+lightglue-official",
    "aliked+lightglue-official",
]
parser = argparse.ArgumentParser()
parser.add_argument("--skip_debug", action="store_true", default=True)
parser.add_argument("--dataset", type=str, default="megadepth",
                    choices=["megadepth", "scannet"],
                    help="Evaluation dataset.")
parser.add_argument("--matcher", type=str, default="superpoint+lightglue-official",
                    choices=_MATCHERS,
                    help="Matcher whose predictions_baseline.h5 to refine.")
parser.add_argument("--cotr_root", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "COTR"),
                    help="Path to the cloned COTR repo.")
parser.add_argument("--cotr_ckpt", type=str, default=None,
                    help="Path to COTR checkpoint. Defaults to <cotr_root>/out/default/checkpoint.pth.tar")
parser.add_argument("--zoom_levels", type=int, default=1,
                    help="Zoom iterations. Sub-pixel precision requires >= 2.")
parser.add_argument("--batch_size", type=int, default=256,
                    help="Number of tasks per SparseEngine batch.")
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

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = standardize_path("./glue-factory")
_DS_FOLDER = f"{args.dataset}1500"
PRED_DIR   = os.path.join(ROOT_DIR, "outputs/results", _DS_FOLDER, args.matcher)
H5_IN      = os.path.join(PRED_DIR, "predictions_baseline.h5")
IMAGE_DIR  = os.path.join(ROOT_DIR, "data", _DS_FOLDER,
                          "images" if args.dataset == "megadepth" else "")

COTR_ROOT  = args.cotr_root
COTR_CKPT  = args.cotr_ckpt or os.path.join(COTR_ROOT, "out", "default", "checkpoint.pth.tar")
ZOOM_LVLS  = args.zoom_levels

# H5 keypoints are in native pixel coordinates (GlueFactory does not resize).
# Images must be loaded at native size so patch extraction coordinates are correct.


# ── COTR model loading ───────────────────────────────────────────────────────
if not os.path.isdir(COTR_ROOT):
    raise FileNotFoundError(f"COTR repo not found at: {COTR_ROOT}")
if not os.path.isfile(COTR_CKPT):
    raise FileNotFoundError(f"COTR checkpoint not found: {COTR_CKPT}")

_saved_path = sys.path[:]
sys.path.insert(0, COTR_ROOT)

# Patch global_configs
_gcfg_path = os.path.join(COTR_ROOT, "COTR", "global_configs", "__init__.py")
with open(_gcfg_path, "r") as _f:
    _gcfg = _f.read()
_ASSERT_OUT   = "assert os.path.isdir(general_config['out']), f'Please create {general_config[\"out\"]}'"
_ASSERT_TB    = "assert os.path.isdir(general_config['tb_out']), f'Please create {general_config[\"tb_out\"]}'"
_MAKEDIRS_OUT = "os.makedirs(general_config['out'], exist_ok=True)"
_MAKEDIRS_TB  = "os.makedirs(general_config['tb_out'], exist_ok=True)"
if _ASSERT_OUT in _gcfg or _ASSERT_TB in _gcfg:
    _gcfg = _gcfg.replace(_ASSERT_OUT, _MAKEDIRS_OUT).replace(_ASSERT_TB, _MAKEDIRS_TB)
    with open(_gcfg_path, "w") as _f:
        _f.write(_gcfg)

# Patch capture.py
_capture_path = os.path.join(COTR_ROOT, "COTR", "cameras", "capture.py")
with open(_capture_path, "r") as _f:
    _capture = _f.read()
_TABLES_HARD = "import tables"
_TABLES_SOFT = ("try:\n    import tables\nexcept ImportError:\n    tables = None")
if _TABLES_HARD in _capture and _TABLES_SOFT not in _capture:
    _capture = _capture.replace(_TABLES_HARD, _TABLES_SOFT)
    with open(_capture_path, "w") as _f:
        _f.write(_capture)

from COTR.models import build_model
from COTR.options.options import set_COTR_arguments
from COTR.inference.sparse_engine import SparseEngine
from COTR.inference.refinement_task import RefinementTask
from COTR.utils.utils import safe_load_weights

sys.path = _saved_path

_cotr_parser = argparse.ArgumentParser()
set_COTR_arguments(_cotr_parser)
_cotr_opt, _ = _cotr_parser.parse_known_args([])
_layer_channels = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
_cotr_opt.dim_feedforward = _layer_channels[_cotr_opt.layer]

cotr_model = build_model(_cotr_opt).to(DEVICE).eval()
_weights = torch.load(COTR_CKPT, map_location=DEVICE, weights_only=False)['model_state_dict']
safe_load_weights(cotr_model, _weights)

engine = SparseEngine(cotr_model, batch_size=args.batch_size, mode='tile')
print(f"[COTR] Loaded from {COTR_CKPT}")


# ── Zoom schedule ────────────────────────────────────────────────────────────
def _zoom_schedule(n: int) -> list[float]:
    return [1.0 / (2 ** i) for i in range(n)]

# ── Crop helpers ─────────────────────────────────────────────────────────────
def _crop_origin(kp_x: float, kp_y: float, H: int, W: int) -> tuple[int, int, int]:
    """Returns (lu_x, lu_y, size) for the local COTR crop centred at (kp_x, kp_y)."""
    size = COTR_MAX_SIZE
    lu_y = int(kp_y) - size // 2
    lu_x = int(kp_x) - size // 2

    # Boundary clamps
    if lu_y < 0: lu_y = 0
    if lu_x < 0: lu_x = 0
    if lu_y + size > H: lu_y = H - size
    if lu_x + size > W: lu_x = W - size

    return lu_x, lu_y, size

# ── Per-pair COTR refinement ─────────────────────────────────────────────────
def cotr_refine_pair(img0_rgb: np.ndarray, img1_rgb: np.ndarray,
                     kp0_px: np.ndarray, kp1_init_px: np.ndarray,
                     zoom_levels: int) -> np.ndarray:
    N = len(kp0_px)
    if N == 0:
        return kp1_init_px.copy()

    if zoom_levels > 1:
        return _cotr_refine_pair_multiscale(img0_rgb, img1_rgb,
                                            kp0_px, kp1_init_px, zoom_levels)

    H0, W0 = img0_rgb.shape[:2]
    H1, W1 = img1_rgb.shape[:2]
    device = next(cotr_model.parameters()).device

    mean_t = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(_IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    origins0 = [_crop_origin(float(kp0_px[i, 0]), float(kp0_px[i, 1]), H0, W0) for i in range(N)]
    origins1 = [_crop_origin(float(kp1_init_px[i, 0]), float(kp1_init_px[i, 1]), H1, W1) for i in range(N)]

    queries_np = np.array(
        [[(float(kp0_px[i, 0]) - origins0[i][0]) / (origins0[i][2] * 2),
          (float(kp0_px[i, 1]) - origins0[i][1]) /  origins0[i][2]]
         for i in range(N)], dtype=np.float32)
    queries_t = torch.from_numpy(queries_np[:, None, :]).to(device)

    refined = kp1_init_px.copy().astype(np.float64)

    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)

        imgs0 = np.stack([
            cv2.resize(
                img0_rgb[origins0[i][1] : origins0[i][1] + origins0[i][2],
                         origins0[i][0] : origins0[i][0] + origins0[i][2]],
                (COTR_MAX_SIZE, COTR_MAX_SIZE), interpolation=cv2.INTER_LINEAR)
            for i in range(start, end)
        ])

        imgs1 = np.stack([
            cv2.resize(
                img1_rgb[origins1[i][1] : origins1[i][1] + origins1[i][2],
                         origins1[i][0] : origins1[i][0] + origins1[i][2]],
                (COTR_MAX_SIZE, COTR_MAX_SIZE), interpolation=cv2.INTER_LINEAR)
            for i in range(start, end)
        ])

        t0 = torch.from_numpy(imgs0).permute(0, 3, 1, 2).float().div(255.0).to(device)
        t1 = torch.from_numpy(imgs1).permute(0, 3, 1, 2).float().div(255.0).to(device)
        t0 = (t0 - mean_t) / std_t
        t1 = (t1 - mean_t) / std_t
        combined = torch.cat([t0, t1], dim=3)

        with torch.no_grad():
            raw_out = cotr_model(combined, queries_t[start:end])['pred_corrs'] \
                          .detach().cpu().numpy()[:, 0, :]

        for j, i in enumerate(range(start, end)):
            lu1_x, lu1_y, sz1 = origins1[i]
            refined[i, 0] = (raw_out[j, 0] - 0.5) * 2 * sz1 + lu1_x
            refined[i, 1] =  raw_out[j, 1]         * sz1 + lu1_y

    return refined.astype(np.float32)


def _cotr_refine_pair_multiscale(img0_rgb: np.ndarray, img1_rgb: np.ndarray,
                                 kp0_px: np.ndarray, kp1_init_px: np.ndarray,
                                 zoom_levels: int) -> np.ndarray:
    N = len(kp0_px)
    zoom_ins = _zoom_schedule(zoom_levels)

    H0, W0 = img0_rgb.shape[:2]
    H1, W1 = img1_rgb.shape[:2]
    area_0 = (COTR_MAX_SIZE * COTR_MAX_SIZE) / float(H0 * W0)
    area_1 = (COTR_MAX_SIZE * COTR_MAX_SIZE) / float(H1 * W1)

    tasks = [
        RefinementTask(
            img0_rgb, img1_rgb,
            kp0_px[i].astype(np.float64),
            kp1_init_px[i].astype(np.float64),
            area_from=area_0,
            area_to=area_1,
            converge_iters=1,
            zoom_ins=zoom_ins,
            identifier=i,
        ) for i in range(N)
    ]

    while True:
        task_ref, img_batch, query_batch = engine.form_batch(tasks)
        if len(task_ref) == 0:
            break

        # FP16 speedup
        # with torch.amp.autocast('cuda'):
        out = engine.infer_batch(img_batch, query_batch)

        for t, o in zip(task_ref, out):
            t.step(o)

    refined = kp1_init_px.copy().astype(np.float64)
    for t in tasks:
        result = t.conclude(force=True)
        if result is not None:
            refined[t.identifier] = result[2:]

    return refined.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    model_name = "cotr"
    os.makedirs(PRED_DIR, exist_ok=True)

    out_files: dict[str, h5py.File] = {}
    final_paths: dict[str, tuple[str, str]] = {}
    thresh_map: dict[str, float] = {}
    for lbl, th in GATE_CONFIGS:
        final_path = os.path.join(PRED_DIR, f"predictions_{model_name}_{lbl}_z{ZOOM_LVLS}.h5")
        tmp_path = final_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        out_files[lbl] = h5py.File(tmp_path, "w")
        final_paths[lbl] = (tmp_path, final_path)
        thresh_map[lbl] = th

    kp0_x_all: list = []
    kp0_y_all: list = []
    native_long_sides: list = []

    with h5py.File(H5_IN, "r") as f_in:
        pairs = [(n0, n1) for n0 in f_in.keys() for n1 in f_in[n0].keys() if "matches0" in f_in[n0][n1]]

        for name0, name1 in tqdm(pairs, desc=f"COTR Local Refinement (z{ZOOM_LVLS})"):
            grp_in = f_in[name0][name1]
            img0_bgr = cv2.imread(os.path.join(IMAGE_DIR, *name0.split("-")))
            img1_bgr = cv2.imread(os.path.join(IMAGE_DIR, *name1.split("-")))

            if img0_bgr is None or img1_bgr is None:
                for f_out in out_files.values():
                    g = f_out.require_group(name0).create_group(name1)
                    for k in grp_in.keys():
                        g.create_dataset(k, data=grp_in[k][:])
                continue

            native_long_sides.append(max(img0_bgr.shape[:2]))
            native_long_sides.append(max(img1_bgr.shape[:2]))

            # No resize: images stay at native size, matching the H5 keypoint frame.

            img0_rgb = img0_bgr[:, :, ::-1].copy()
            img1_rgb = img1_bgr[:, :, ::-1].copy()

            kp0 = grp_in["keypoints0"][:]
            kp1 = grp_in["keypoints1"][:]
            m0  = grp_in["matches0"][:]

            kp0_x_all.append(kp0[:, 0])
            kp0_y_all.append(kp0[:, 1])

            idx0        = np.where(m0 > -1)[0]
            matched_kp0 = kp0[idx0]
            matched_kp1 = kp1[m0[idx0]]

            if len(matched_kp0) == 0:
                for f_out in out_files.values():
                    g = f_out.require_group(name0).create_group(name1)
                    for k in grp_in.keys():
                        g.create_dataset(k, data=grp_in[k][:])
                continue

            refined_kp1 = cotr_refine_pair(img0_rgb, img1_rgb,
                                           matched_kp0, matched_kp1,
                                           zoom_levels=ZOOM_LVLS)
            deltas = refined_kp1 - matched_kp1

            # Apply gating & write H5
            for lbl, f_out in out_files.items():
                current_kp1     = kp1.copy()
                current_matched = matched_kp1.copy()

                mags  = np.linalg.norm(deltas, axis=1)
                valid = mags <= thresh_map[lbl]

                current_matched[valid] = refined_kp1[valid]
                current_kp1[m0[idx0]] = current_matched

                g = f_out.require_group(name0).create_group(name1)
                for k in grp_in.keys():
                    data = current_kp1 if k == "keypoints1" else grp_in[k][:]
                    g.create_dataset(k, data=data)

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
            logf.write(f"# COTR coord-space sanity check\n")
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
        print(f"  kp0_x max={kp0_x.max():.1f}, kp0_y max={kp0_y.max():.1f} (expected ~{expected_max})")

    print("\nDone.")

if __name__ == "__main__":
    main()