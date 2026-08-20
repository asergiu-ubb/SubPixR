"""Benchmark GPU inference time of refiners on dummy data.

Times each refiner's cost to "refine" a fixed batch of query points
on a single 480x640 image pair, repeated R=100 timed iterations after
W=20 warmup iterations. Uses torch.cuda.Event for accurate timing.

Refiners benchmarked:
  - SubPixR (ours)              : sparse, batched per-keypoint
  - S2DNet                      : sparse, hierarchical pyramid
  - CAPS                        : sparse, batched per-keypoint
  - Patch2Pix                   : per-pair (not per-keypoint)
  - COTR z=1                    : sparse, single-scale local
  - COTR z=2                    : sparse, multi-scale recursive
  - LoFTR                       : per-pair (matcher; produces own kps)
  - Lucas-Kanade (CPU baseline) : classical, per-keypoint
  - NCC Matching (CPU baseline) : classical, per-keypoint
  - Dense SIFT (CPU baseline)   : classical, per-keypoint
  - Dense ORB (CPU baseline)    : classical, per-keypoint

Output CSV columns:
  method, params_M, ms_setup_per_pair, ms_per_keypoint, ms_total_1kp, ms_total_2kp, ...
  granularity, device, notes

Run from this directory:
  cd eval_sota_with_refiner
  python benchmark_refiner_speed.py
"""
import os
import sys
import csv
import time
import argparse
import traceback
import numpy as np
import torch

# All paths resolve relative to THIS file, not the cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_SCRIPT_DIR, '..')))
sys.path.append(_SCRIPT_DIR)

from subpixr.utils import standardize_path

_GF_ROOT  = standardize_path("./glue-factory")
_MATCHER  = "superpoint+lightglue-official"

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
KEYPOINT_COUNTS = [1, 2, 4, 8, 10, 16, 32, 64, 96, 128, 192, 256, 448, 512]
IMG_H, IMG_W = 480, 640
WARMUP = 4
REPEATS = 10
REFINER_BATCH = 512          # batch size for sparse-per-keypoint methods
OUTPUT_CSV = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'paper_refiner', 'inference_speed.csv'))

# ── Toggle individual benchmarks here ─────────────────────────────────────
ACTIVE_BENCHMARKS = {
    'SubPixR (ours)':                    False,   # hpatches: 1 forward pass, batch of N crops
    'S2DNet':                            False,
    'CAPS':                              False,
    'Patch2Pix':                         False,
    'COTR (z=1)':                        False,  # legacy single-scale crop
    'COTR (z=2)':                        False,  # legacy multi-scale full-frame
    'COTR (z=3, full-frame)':            False,  # generic 480x640 — use dataset-specific below
    'COTR (z=3, full-frame, megadepth)': False,  # mean size probed from megadepth1500 H5
    'COTR (z=3, full-frame, scannet)':   False,  # mean size probed from scannet1500 H5
    'COTR (z=3, patch-128)':             False,   # hpatches mode: 3 sequential batches per kp
    'LoFTR':                             False,
    'Lucas-Kanade (CPU)':                False,
    'NCC Matching (CPU)':                False,
    'Dense SIFT (CPU)':                  True,
    'Dense ORB (CPU)':                   True,
}

# ─────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────
def _time_cuda(fn, warmup=WARMUP, repeats=REPEATS):
    """Run fn() warmup times, then time it repeats times. Returns ms (mean)."""
    for _ in range(warmup):
        fn()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_start.record()
        for _ in range(repeats):
            fn()
        e_end.record()
        torch.cuda.synchronize()
        return e_start.elapsed_time(e_end) / repeats
    else:
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t0) / repeats * 1000.0


def _count_params(model):
    if model is None:
        return None
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def _probe_dataset_mean_size(dataset: str, max_sample: int = 100):
    """Parse the baseline H5 for *dataset* ('megadepth'|'scannet'), open up to
    *max_sample* unique image headers with PIL, and return (mean_H, mean_W) as ints."""
    import h5py
    from PIL import Image as _PIL

    ds_folder = f"{dataset}1500"
    h5_path   = os.path.join(_GF_ROOT, "outputs", "results", ds_folder,
                             _MATCHER, "predictions_baseline.h5")
    img_dir   = os.path.join(_GF_ROOT, "data", ds_folder,
                             "images" if dataset == "megadepth" else "")

    def _key_to_path(key):
        return os.path.join(img_dir, *key.split('-'))

    unique_keys: set = set()
    with h5py.File(h5_path, 'r') as f:
        for n0 in f.keys():
            for n1 in f[n0].keys():
                unique_keys.add(n0)
                unique_keys.add(n1)

    keys_list = sorted(unique_keys)
    # Evenly sample up to max_sample keys so we don't read thousands of files
    if len(keys_list) > max_sample:
        idx = np.linspace(0, len(keys_list) - 1, max_sample).round().astype(int)
        keys_list = [keys_list[i] for i in idx]

    print(f"[probe] {dataset}: sampling {len(keys_list)} / {len(unique_keys)} unique images...",
          flush=True)

    Hs, Ws = [], []
    for i, k in enumerate(keys_list):
        try:
            with _PIL.open(_key_to_path(k)) as im:
                w, h = im.size
            Hs.append(h); Ws.append(w)
        except Exception:
            pass
        if (i + 1) % 20 == 0 or (i + 1) == len(keys_list):
            print(f"[probe]   {i+1}/{len(keys_list)} read...", flush=True)

    mean_h = int(round(float(np.mean(Hs))))
    mean_w = int(round(float(np.mean(Ws))))
    print(f"[probe] {dataset}: mean H={mean_h}  mean W={mean_w}  "
          f"(median H={int(np.median(Hs))}  W={int(np.median(Ws))})", flush=True)
    return mean_h, mean_w


def _make_dummy_pair(n_keypoints, img_h: int = IMG_H, img_w: int = IMG_W):
    """Returns (img1_uint8, img2_uint8, kpts1, kpts2) — img is HxWx3 uint8 RGB,
    kpts is float32 [N, 2] uniformly inside a margin."""
    rng = np.random.RandomState(0)
    img1 = rng.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8)
    img2 = rng.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8)
    margin = 80
    xs = rng.randint(margin, img_w - margin, n_keypoints).astype(np.float32)
    ys = rng.randint(margin, img_h - margin, n_keypoints).astype(np.float32)
    kpts1 = np.stack([xs, ys], axis=1)
    kpts2 = kpts1 + rng.uniform(-5, 5, kpts1.shape).astype(np.float32)
    return img1, img2, kpts1, kpts2


# ─────────────────────────────────────────────────────────────────────────
# Per-refiner benchmark functions. Each returns a dict (a CSV row).
# ─────────────────────────────────────────────────────────────────────────
def bench_subpixr():
    from subpixr.utils import get_general_refiner, load_refiner
    weights = get_general_refiner()
    model, scale_factor = load_refiner(weights, device=DEVICE)
    model.eval()
    n_params = _count_params(model)

    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    def _to_t(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        return (t - norm_mean) / norm_std

    res = {
        'method': 'SubPixR (ours)',
        'params_M': f"{n_params:.2f}",
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse',
        'device': DEVICE,
        'notes': f'single-pass forward, batched {REFINER_BATCH}',
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, kpts2 = _make_dummy_pair(n_kp)
        ref_t = _to_t(img1)
        trg_t = _to_t(img2)
        pts1_t = torch.from_numpy(kpts1).to(DEVICE)
        pts2_t = torch.from_numpy(kpts2).to(DEVICE)

        @torch.no_grad()
        def step():
            for b in range(0, n_kp, REFINER_BATCH):
                end = min(b + REFINER_BATCH, n_kp)
                bs = end - b
                tl_1 = torch.stack([torch.round(pts1_t[b:end, 0]) - 64.0,
                                    torch.round(pts1_t[b:end, 1]) - 64.0], dim=1)
                tl_2 = torch.stack([torch.round(pts2_t[b:end, 0]) - 64.0,
                                    torch.round(pts2_t[b:end, 1]) - 64.0], dim=1)
                t_crops = model.extract_patch(ref_t.expand(bs, -1, -1, -1), tl_1, 128)
                s_crops = model.extract_patch(trg_t.expand(bs, -1, -1, -1), tl_2, 128)
                _ = model(t_crops, s_crops, num_iters=1)

        ms = _time_cuda(step)
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_s2dnet():
    from s2dnet_refiner import load_s2dnet, extract_pyramid, refine_hierarchical
    _S2D_ROOT = os.path.join(_SCRIPT_DIR, 'S2DNet-Minimal')
    _S2D_CKPT = os.path.join(_S2D_ROOT, 's2dnet_weights.pth')
    if not os.path.isfile(_S2D_CKPT):
        raise FileNotFoundError(f"S2DNet checkpoint not found: {_S2D_CKPT}")
    model, level_scales = load_s2dnet(_S2D_CKPT, DEVICE)
    n_params = _count_params(model)

    res = {
        'method': 'S2DNet',
        'params_M': f"{n_params:.2f}" if n_params else '?',
        'granularity': 'sparse (hierarchical)',
        'device': DEVICE,
        'notes': 'pyramid extracted once per pair, then per-kp refine',
    }

    per_kp_times = []
    # Setup is done once per pair
    img1, img2, _, _ = _make_dummy_pair(1)
    def _setup():
        ref_pyr = extract_pyramid(img1, model, DEVICE)
        trg_pyr = extract_pyramid(img2, model, DEVICE)
        return ref_pyr, trg_pyr
    setup_ms = _time_cuda(lambda: _setup(), repeats=20)
    res['ms_setup_per_pair'] = f"{setup_ms:.2f}"

    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, kpts2 = _make_dummy_pair(n_kp)
        ref_pyr, trg_pyr = _setup()
        def step():
            _, _ = refine_hierarchical(ref_pyr, trg_pyr,
                                       kpts1.astype(np.float32),
                                       kpts2.astype(np.float32),
                                       level_scales)
        refine_ms = _time_cuda(step, repeats=20)
        total_ms = setup_ms + refine_ms
        res[f'ms_total_{n_kp}kp'] = f"{total_ms:.2f}"
        per_kp_times.append(refine_ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_caps():
    _caps_root = os.path.join(_SCRIPT_DIR, 'caps')
    _caps_ckpt = os.path.join(_caps_root, 'weights', 'caps-pretrained.pth')
    if not os.path.isfile(_caps_ckpt):
        raise FileNotFoundError(f"CAPS checkpoint not found: {_caps_ckpt}")
    _saved = sys.path[:]
    sys.path.insert(0, os.path.join(_caps_root, 'CAPS'))
    try:
        from network import CAPSNet
    finally:
        sys.path = _saved
    import types
    args = types.SimpleNamespace(
        pretrained=1, backbone='resnet50', coarse_feat_dim=128, fine_feat_dim=128,
        prob_from='correlation', use_nn=1, window_size=0.125,
    )
    model = CAPSNet(args, DEVICE).eval().to(DEVICE)
    ckpt = torch.load(_caps_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    n_params = _count_params(model)

    res = {
        'method': 'CAPS',
        'params_M': f"{n_params:.2f}",
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse',
        'device': DEVICE,
        'notes': 'all kpts in single forward; ResNet-50 backbone',
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, _ = _make_dummy_pair(n_kp)
        t1 = torch.from_numpy(img1).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0
        t2 = torch.from_numpy(img2).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0
        kp = torch.from_numpy(kpts1)[None].to(DEVICE)

        @torch.no_grad()
        def step():
            _, _ = model.test(t1, t2, kp)

        ms = _time_cuda(step)
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_patch2pix():
    import tempfile, cv2
    _p2p_root = os.path.join(_SCRIPT_DIR, 'patch2pix')
    _p2p_ckpt = os.path.join(_p2p_root, 'pretrained', 'patch2pix_pretrained.pth')
    if not os.path.isfile(_p2p_ckpt):
        raise FileNotFoundError(f"Patch2Pix checkpoint not found: {_p2p_ckpt}")
    _saved = sys.path[:]
    sys.path = [_p2p_root] + [p for p in sys.path if p not in ('', '.', '..')]
    try:
        from utils.eval.model_helper import load_model as p2p_load
        from utils.eval.model_helper import estimate_matches as p2p_estimate
    finally:
        sys.path = _saved
    model = p2p_load(_p2p_ckpt, method='patch2pix')

    img1, img2, _, _ = _make_dummy_pair(1)
    f1 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    f2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(f1.name, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)); f1.close()
    cv2.imwrite(f2.name, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)); f2.close()

    n_params = None
    try:
        n_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad) / 1e6
    except Exception:
        pass

    def step():
        _, _, _ = p2p_estimate(model, f1.name, f2.name, ksize=2,
                               io_thres=0.25, eval_type='fine', imsize=1024)
    ms = _time_cuda(step, repeats=20)
    os.unlink(f1.name); os.unlink(f2.name)

    res = {
        'method': 'Patch2Pix',
        'params_M': f"{n_params:.2f}" if n_params else '?',
        'ms_setup_per_pair': f"{ms:.2f}",
        'ms_per_keypoint': '—',
        'granularity': 'per-pair',
        'device': DEVICE,
        'notes': 'imsize=1024; matcher+refiner, produces own match set',
    }

    # Patch2Pix does not take a specific number of keypoints, it processes the whole image
    for n_kp in KEYPOINT_COUNTS:
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"

    return res


# ─────────────────────────────────────────────────────────────────────────
# COTR — single load, two zoom modes (z=1 single-scale, z=2 multi-scale)
# ─────────────────────────────────────────────────────────────────────────
_COTR_CACHE = {'model': None, 'engine': None, 'params': None}

def _load_cotr():
    """Lazy-load COTR once and return (model, engine, n_params_M)."""
    if _COTR_CACHE['model'] is not None:
        return _COTR_CACHE['model'], _COTR_CACHE['engine'], _COTR_CACHE['params']

    _COTR_ROOT = os.path.join(_SCRIPT_DIR, 'COTR')
    _COTR_CKPT = os.path.join(_COTR_ROOT, 'out', 'default', 'checkpoint.pth.tar')
    if not os.path.isdir(_COTR_ROOT):
        raise FileNotFoundError(f"COTR repo not found: {_COTR_ROOT}")
    if not os.path.isfile(_COTR_CKPT):
        raise FileNotFoundError(f"COTR checkpoint not found: {_COTR_CKPT}")

    _saved = sys.path[:]
    sys.path.insert(0, _COTR_ROOT)

    # Patch global_configs to not assert on missing dirs (matches eval_COTR_sota.py).
    _gcfg_path = os.path.join(_COTR_ROOT, 'COTR', 'global_configs', '__init__.py')
    if os.path.isfile(_gcfg_path):
        with open(_gcfg_path, 'r') as _f:
            _gcfg = _f.read()
        _ASSERT_OUT = "assert os.path.isdir(general_config['out']), f'Please create {general_config[\"out\"]}'"
        _ASSERT_TB = "assert os.path.isdir(general_config['tb_out']), f'Please create {general_config[\"tb_out\"]}'"
        _MAKEDIRS_OUT = "os.makedirs(general_config['out'], exist_ok=True)"
        _MAKEDIRS_TB = "os.makedirs(general_config['tb_out'], exist_ok=True)"
        if _ASSERT_OUT in _gcfg or _ASSERT_TB in _gcfg:
            _gcfg = _gcfg.replace(_ASSERT_OUT, _MAKEDIRS_OUT).replace(_ASSERT_TB, _MAKEDIRS_TB)
            with open(_gcfg_path, 'w') as _f:
                _f.write(_gcfg)

    # Patch capture.py to make pytables import optional.
    _capture_path = os.path.join(_COTR_ROOT, 'COTR', 'cameras', 'capture.py')
    if os.path.isfile(_capture_path):
        with open(_capture_path, 'r') as _f:
            _capture = _f.read()
        _TABLES_HARD = "import tables"
        _TABLES_SOFT = "try:\n    import tables\nexcept ImportError:\n    tables = None"
        if _TABLES_HARD in _capture and _TABLES_SOFT not in _capture:
            _capture = _capture.replace(_TABLES_HARD, _TABLES_SOFT)
            with open(_capture_path, 'w') as _f:
                _f.write(_capture)

    from COTR.models import build_model
    from COTR.options.options import set_COTR_arguments
    from COTR.inference.sparse_engine import SparseEngine
    from COTR.utils.utils import safe_load_weights

    sys.path = _saved

    _cotr_parser = argparse.ArgumentParser()
    set_COTR_arguments(_cotr_parser)
    _cotr_opt, _ = _cotr_parser.parse_known_args([])
    _layer_channels = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
    _cotr_opt.dim_feedforward = _layer_channels[_cotr_opt.layer]

    model = build_model(_cotr_opt).to(DEVICE).eval()
    weights = torch.load(_COTR_CKPT, map_location=DEVICE, weights_only=False)['model_state_dict']
    safe_load_weights(model, weights)
    engine = SparseEngine(model, batch_size=REFINER_BATCH, mode='tile')
    n_params = _count_params(model)

    _COTR_CACHE['model'] = model
    _COTR_CACHE['engine'] = engine
    _COTR_CACHE['params'] = n_params
    return model, engine, n_params


_COTR_MAX_SIZE = 256
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _cotr_crop_origin(kp_x, kp_y, H, W):
    size = _COTR_MAX_SIZE
    lu_y = int(kp_y) - size // 2
    lu_x = int(kp_x) - size // 2
    if lu_y < 0: lu_y = 0
    if lu_x < 0: lu_x = 0
    if lu_y + size > H: lu_y = H - size
    if lu_x + size > W: lu_x = W - size
    return lu_x, lu_y, size


def bench_cotr_z1():
    """COTR single-scale local refinement (z=1)."""
    import cv2
    model, _, n_params = _load_cotr()
    device = next(model.parameters()).device

    mean_t = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    res = {
        'method': 'COTR (z=1)',
        'params_M': f"{n_params:.2f}",
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse',
        'device': DEVICE,
        'notes': 'single-scale 256x256 local crop refinement',
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, kpts2 = _make_dummy_pair(n_kp)
        H0, W0 = img1.shape[:2]; H1, W1 = img2.shape[:2]

        origins0 = [_cotr_crop_origin(float(kpts1[i, 0]), float(kpts1[i, 1]), H0, W0) for i in range(n_kp)]
        origins1 = [_cotr_crop_origin(float(kpts2[i, 0]), float(kpts2[i, 1]), H1, W1) for i in range(n_kp)]

        queries_np = np.array(
            [[(float(kpts1[i, 0]) - origins0[i][0]) / (origins0[i][2] * 2),
              (float(kpts1[i, 1]) - origins0[i][1]) / origins0[i][2]]
             for i in range(n_kp)], dtype=np.float32)
        queries_t = torch.from_numpy(queries_np[:, None, :]).to(device)

        @torch.no_grad()
        def step():
            for start in range(0, n_kp, REFINER_BATCH):
                end = min(start + REFINER_BATCH, n_kp)
                imgs0 = np.stack([
                    cv2.resize(img1[origins0[i][1]:origins0[i][1]+origins0[i][2],
                                    origins0[i][0]:origins0[i][0]+origins0[i][2]],
                               (_COTR_MAX_SIZE, _COTR_MAX_SIZE), interpolation=cv2.INTER_LINEAR)
                    for i in range(start, end)
                ])
                imgs1 = np.stack([
                    cv2.resize(img2[origins1[i][1]:origins1[i][1]+origins1[i][2],
                                    origins1[i][0]:origins1[i][0]+origins1[i][2]],
                               (_COTR_MAX_SIZE, _COTR_MAX_SIZE), interpolation=cv2.INTER_LINEAR)
                    for i in range(start, end)
                ])
                t0 = torch.from_numpy(imgs0).permute(0, 3, 1, 2).float().div(255.0).to(device)
                t1 = torch.from_numpy(imgs1).permute(0, 3, 1, 2).float().div(255.0).to(device)
                t0 = (t0 - mean_t) / std_t
                t1 = (t1 - mean_t) / std_t
                combined = torch.cat([t0, t1], dim=3)
                _ = model(combined, queries_t[start:end])['pred_corrs']

        ms = _time_cuda(step, repeats=20)
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_cotr_z2():
    """COTR multi-scale recursive refinement (z=2)."""
    model, engine, n_params = _load_cotr()
    _saved = sys.path[:]
    sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'COTR'))
    try:
        from COTR.inference.refinement_task import RefinementTask
    finally:
        sys.path = _saved

    res = {
        'method': 'COTR (z=2)',
        'params_M': f"{n_params:.2f}",
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse (multi-scale)',
        'device': DEVICE,
        'notes': 'recursive zoom (zoom_ins=[1.0, 0.5]) via SparseEngine',
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, kpts2 = _make_dummy_pair(n_kp)
        H0, W0 = img1.shape[:2]; H1, W1 = img2.shape[:2]
        area_0 = (_COTR_MAX_SIZE * _COTR_MAX_SIZE) / float(H0 * W0)
        area_1 = (_COTR_MAX_SIZE * _COTR_MAX_SIZE) / float(H1 * W1)
        zoom_ins = [1.0, 0.5]   # z=2: full + half-zoom

        def step():
            tasks = [
                RefinementTask(
                    img1, img2,
                    kpts1[i].astype(np.float64),
                    kpts2[i].astype(np.float64),
                    area_from=area_0, area_to=area_1,
                    converge_iters=1, zoom_ins=zoom_ins, identifier=i,
                ) for i in range(n_kp)
            ]
            while True:
                task_ref, img_batch, query_batch = engine.form_batch(tasks)
                if len(task_ref) == 0:
                    break
                with torch.amp.autocast('cuda'):
                    out = engine.infer_batch(img_batch, query_batch)
                for t, o in zip(task_ref, out):
                    t.step(o)
            for t in tasks:
                t.conclude(force=True)

        ms = _time_cuda(step, warmup=3, repeats=10)
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_cotr_z3_fullframe(img_h: int = IMG_H, img_w: int = IMG_W,
                             method_label: str = 'COTR (z=3, full-frame)'):
    """COTR z=3, full-frame — mirrors eval_COTR_native.py / megadepth-scannet mode.
    img_h / img_w default to the script-level IMG_H/IMG_W but can be overridden
    with the mean dimensions measured from MegaDepth or ScanNet."""
    model, engine, n_params = _load_cotr()
    _saved = sys.path[:]
    sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'COTR'))
    try:
        from COTR.inference.refinement_task import RefinementTask
    finally:
        sys.path = _saved

    zoom_ins = np.linspace(0.5, 0.0625, 3)

    res = {
        'method': method_label,
        'params_M': f"{n_params:.2f}",
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse (multi-scale)',
        'device': DEVICE,
        'notes': f'z=3 linspace zoom on full {img_h}x{img_w} image — matches megadepth/scannet eval',
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        print(f"  [{method_label}] n_kp={n_kp} — warmup+timing ...", flush=True)
        img1, img2, kpts1, kpts2 = _make_dummy_pair(n_kp, img_h, img_w)
        H0, W0 = img1.shape[:2]
        H1, W1 = img2.shape[:2]
        area_0 = (_COTR_MAX_SIZE ** 2) / float(H0 * W0)
        area_1 = (_COTR_MAX_SIZE ** 2) / float(H1 * W1)

        def step():
            tasks = [
                RefinementTask(img1, img2,
                               kpts1[i].astype(np.float64), kpts2[i].astype(np.float64),
                               area_from=area_0, area_to=area_1,
                               converge_iters=1, zoom_ins=zoom_ins, identifier=i)
                for i in range(n_kp)
            ]
            while True:
                task_ref, img_batch, query_batch = engine.form_batch(tasks)
                if len(task_ref) == 0:
                    break
                with torch.no_grad():
                    out = engine.infer_batch(img_batch, query_batch)
                for t, o in zip(task_ref, out):
                    t.step(o)
            for t in tasks:
                t.conclude(force=True)

        ms = _time_cuda(step, warmup=3, repeats=10)
        print(f"  [{method_label}] n_kp={n_kp}  -> {ms:.1f} ms", flush=True)
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_cotr_z3_fullframe_megadepth():
    """COTR z=3, full-frame at MegaDepth mean image size (probed from H5)."""
    h, w = _probe_dataset_mean_size('megadepth')
    return bench_cotr_z3_fullframe(img_h=h, img_w=w,
                                   method_label=f'COTR (z=3, full-frame, megadepth {h}x{w})')


def bench_cotr_z3_fullframe_scannet():
    """COTR z=3, full-frame at ScanNet mean image size (probed from H5)."""
    h, w = _probe_dataset_mean_size('scannet')
    return bench_cotr_z3_fullframe(img_h=h, img_w=w,
                                   method_label=f'COTR (z=3, full-frame, scannet {h}x{w})')


def bench_cotr_z3_patch128():
    """COTR z=3, 128x128 crop per keypoint — mirrors hpatches_refiner_MMA.py mode."""
    model, engine, n_params = _load_cotr()
    _saved = sys.path[:]
    sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'COTR'))
    try:
        from COTR.inference.refinement_task import RefinementTask
    finally:
        sys.path = _saved

    zoom_ins = np.linspace(0.5, 0.0625, 3)
    half = 64  # 128px crop → half = 64

    def _crop_patch(img, cx, cy):
        H, W = img.shape[:2]
        x0 = max(0, min(W - 2 * half, int(round(cx)) - half))
        y0 = max(0, min(H - 2 * half, int(round(cy)) - half))
        return img[y0:y0 + 2 * half, x0:x0 + 2 * half], x0, y0

    res = {
        'method': 'COTR (z=3, patch-128)',
        'params_M': f"{n_params:.2f}",
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse (multi-scale)',
        'device': DEVICE,
        'notes': 'z=3 linspace zoom on 128x128 crop per keypoint — matches hpatches eval',
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, kpts2 = _make_dummy_pair(n_kp)
        crop_area = (_COTR_MAX_SIZE ** 2) / float(2 * half * 2 * half)  # 256²/128² = 4.0

        def step():
            tasks = []
            for i in range(n_kp):
                c0, ox0, oy0 = _crop_patch(img1, kpts1[i, 0], kpts1[i, 1])
                c1, ox1, oy1 = _crop_patch(img2, kpts2[i, 0], kpts2[i, 1])
                q0 = np.array([kpts1[i, 0] - ox0, kpts1[i, 1] - oy0], dtype=np.float64)
                q1 = np.array([kpts2[i, 0] - ox1, kpts2[i, 1] - oy1], dtype=np.float64)
                tasks.append(RefinementTask(c0, c1, q0, q1,
                                            area_from=crop_area, area_to=crop_area,
                                            converge_iters=1, zoom_ins=zoom_ins, identifier=i))
            while True:
                task_ref, img_batch, query_batch = engine.form_batch(tasks)
                if len(task_ref) == 0:
                    break
                with torch.no_grad():
                    out = engine.infer_batch(img_batch, query_batch)
                for t, o in zip(task_ref, out):
                    t.step(o)
            for t in tasks:
                t.conclude(force=True)

        ms = _time_cuda(step, warmup=3, repeats=10)
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_loftr():
    import kornia.feature as KF
    import cv2
    model = KF.LoFTR(pretrained='outdoor').eval().to(DEVICE)
    n_params = _count_params(model)

    img1, img2, _, _ = _make_dummy_pair(1)
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    t1 = torch.from_numpy(g1).float()[None, None].to(DEVICE) / 255.0
    t2 = torch.from_numpy(g2).float()[None, None].to(DEVICE) / 255.0

    @torch.no_grad()
    def step():
        _ = model({'image0': t1, 'image1': t2})

    ms = _time_cuda(step)

    res = {
        'method': 'LoFTR',
        'params_M': f"{n_params:.2f}",
        'ms_setup_per_pair': f"{ms:.2f}",
        'ms_per_keypoint': '—',
        'granularity': 'per-pair',
        'device': DEVICE,
        'notes': 'semi-dense matcher, not strictly a refiner',
    }

    for n_kp in KEYPOINT_COUNTS:
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"

    return res


def bench_lucas_kanade():
    import cv2
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    res = {
        'method': 'Lucas-Kanade (CPU)',
        'params_M': '0.00',
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse',
        'device': 'cpu',
        'notes': 'OpenCV pyrLK, win=21, maxLevel=3',
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, _ = _make_dummy_pair(n_kp)
        g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        p0 = kpts1.reshape(-1, 1, 2).astype(np.float32)

        def step():
            _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, **lk_params)

        for _ in range(10):
            step()
        t0 = time.perf_counter()
        for _ in range(50):
            step()
        ms = (time.perf_counter() - t0) / 50 * 1000.0
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_ncc():
    import cv2
    PSZ, SSZ = 31, 51

    def _crop(img, cx, cy, sz):
        x = int(round(cx - sz // 2)); y = int(round(cy - sz // 2))
        return img[y:y + sz, x:x + sz]

    res = {
        'method': 'NCC Matching (CPU)',
        'params_M': '0.00',
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse',
        'device': 'cpu',
        'notes': f'OpenCV TM_CCORR_NORMED, template={PSZ}x{PSZ}, search={SSZ}x{SSZ}',
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, kpts2 = _make_dummy_pair(n_kp)
        g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        def step():
            for k in range(n_kp):
                t = _crop(g1, kpts1[k, 0], kpts1[k, 1], PSZ)
                s = _crop(g2, kpts2[k, 0], kpts2[k, 1], SSZ)
                if t.shape != (PSZ, PSZ) or s.shape != (SSZ, SSZ):
                    continue
                _ = cv2.matchTemplate(s, t, cv2.TM_CCORR_NORMED)

        t0 = time.perf_counter()
        for _ in range(20):
            step()
        ms = (time.perf_counter() - t0) / 20 * 1000.0
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def _bench_dense_feature(name, extractor, matcher, step=2, desc=''):
    """Helper for dense SIFT and dense ORB matching benchmarks."""
    import cv2
    PSZ, SSZ = 64, 128  # 64x64 template and 128x128 search region

    def _crop(img, cx, cy, sz):
        x = int(round(cx - sz // 2)); y = int(round(cy - sz // 2))
        H, W = img.shape[:2]
        if x < 0: x = 0
        if y < 0: y = 0
        if x + sz > W: x = W - sz
        if y + sz > H: y = H - sz
        return img[y:y + sz, x:x + sz]

    res = {
        'method': name,
        'params_M': '0.00',
        'ms_setup_per_pair': '0.0',
        'granularity': 'sparse',
        'device': 'cpu',
        'notes': desc,
    }

    per_kp_times = []
    for n_kp in KEYPOINT_COUNTS:
        img1, img2, kpts1, kpts2 = _make_dummy_pair(n_kp)
        g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        def step_fn():
            for k in range(n_kp):
                t = _crop(g1, kpts1[k, 0], kpts1[k, 1], PSZ)
                s = _crop(g2, kpts2[k, 0], kpts2[k, 1], SSZ)
                if t.shape[0] != PSZ or s.shape[0] != SSZ:
                    continue

                kp_t = [cv2.KeyPoint(PSZ / 2.0, PSZ / 2.0, 15.0)]
                _, des_t = extractor.compute(t, kp_t)
                if des_t is None: continue

                kp_s = [cv2.KeyPoint(float(x), float(y), 15.0)
                        for y in range(0, SSZ, step) for x in range(0, SSZ, step)]
                _, des_s = extractor.compute(s, kp_s)
                if des_s is None: continue

                matches = matcher.match(des_t, des_s)
                if matches:
                    _ = min(matches, key=lambda x: x.distance)

        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            step_fn()
        ms = (time.perf_counter() - t0) / iters * 1000.0
        res[f'ms_total_{n_kp}kp'] = f"{ms:.2f}"
        per_kp_times.append(ms / n_kp)

    res['ms_per_keypoint'] = f"{np.median(per_kp_times):.4f}"
    return res


def bench_dense_sift():
    import cv2
    extractor = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return _bench_dense_feature('Dense SIFT (CPU)', extractor, matcher, step=2,
                                desc='OpenCV SIFT, template=64x64, search=128x128, step=2')


def bench_dense_orb():
    import cv2
    extractor = cv2.ORB_create(edgeThreshold=0, patchSize=15)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return _bench_dense_feature('Dense ORB (CPU)', extractor, matcher, step=2,
                                desc='OpenCV ORB, template=64x64, search=128x128, step=2')


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────
BENCHMARKS = [
    ('SubPixR (ours)',                    bench_subpixr),
    ('S2DNet',                            bench_s2dnet),
    ('CAPS',                              bench_caps),
    ('Patch2Pix',                         bench_patch2pix),
    ('COTR (z=1)',                        bench_cotr_z1),
    ('COTR (z=2)',                        bench_cotr_z2),
    ('COTR (z=3, full-frame)',            bench_cotr_z3_fullframe),
    ('COTR (z=3, full-frame, megadepth)', bench_cotr_z3_fullframe_megadepth),
    ('COTR (z=3, full-frame, scannet)',   bench_cotr_z3_fullframe_scannet),
    ('COTR (z=3, patch-128)',             bench_cotr_z3_patch128),
    ('LoFTR',                             bench_loftr),
    ('Lucas-Kanade (CPU)',                bench_lucas_kanade),
    ('NCC Matching (CPU)',                bench_ncc),
    ('Dense SIFT (CPU)',                  bench_dense_sift),
    ('Dense ORB (CPU)',                   bench_dense_orb),
]


def main():
    print(f"\n[bench] device={DEVICE}  warmup={WARMUP}  repeats={REPEATS}")
    if DEVICE == 'cuda':
        print(f"[bench] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[bench] script_dir={_SCRIPT_DIR}")
    print()

    rows = []
    for name, fn in BENCHMARKS:
        if not ACTIVE_BENCHMARKS.get(name, True):
            print(f"[bench] {name}  [skipped — set ACTIVE_BENCHMARKS['{name}']=True to enable]")
            continue
        print(f"[bench] {name}...", flush=True)
        try:
            row = fn()
            print(f"        params={row['params_M']}M  setup={row['ms_setup_per_pair']}ms  "
                  f"per_kp={row['ms_per_keypoint']}ms")
            for n_kp in KEYPOINT_COUNTS:
                print(f"        total_{n_kp}kp={row[f'ms_total_{n_kp}kp']}ms")
            rows.append(row)
        except Exception as e:
            print(f"        [SKIP] {type(e).__name__}: {e}")
            traceback.print_exc()
            error_row = {
                'method': name,
                'params_M': 'n/a', 'ms_setup_per_pair': 'n/a',
                'ms_per_keypoint': 'n/a',
                'granularity': 'n/a',
                'device': DEVICE,
                'notes': f'failed: {type(e).__name__}: {str(e)[:80]}',
            }
            for n_kp in KEYPOINT_COUNTS:
                error_row[f'ms_total_{n_kp}kp'] = 'n/a'
            rows.append(error_row)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    cols = ['method', 'params_M', 'ms_setup_per_pair', 'ms_per_keypoint']
    for n_kp in KEYPOINT_COUNTS:
        cols.append(f'ms_total_{n_kp}kp')
    cols.extend(['granularity', 'device', 'notes'])

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[bench] wrote {OUTPUT_CSV}")


if __name__ == '__main__':
    main()