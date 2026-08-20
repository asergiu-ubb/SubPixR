"""Utility helpers shared across the SubPixR training and evaluation scripts.

Paths are configurable via environment variables so the same code runs on any
machine without editing source. Defaults point at typical mount layouts used
in the original development setup.
"""
import json
import os
import sys
from sys import platform

import cv2
import numpy as np
import torch
from torchvision import transforms


def standardize_path(path: str) -> str:
    """Pass-through path normaliser. Kept as a hook so external mounts can be
    rewritten cross-platform without touching call sites. By default just
    normalises backslashes to forward slashes.
    """
    return path.replace("\\", "/")


def change_path_for_linux(path: str) -> str:
    """Identity helper retained for backwards compatibility with call sites
    that need a single function to massage paths across platforms.
    """
    return path.replace("\\", "/")


def get_general_refiner() -> str:
    """Path to the released SubPixR checkpoint (best_model.pth).

    Ships with the repo at checkpoints/best_model.pth; override with the
    SUBPIXR_CHECKPOINT environment variable.
    """
    env = os.environ.get("SUBPIXR_CHECKPOINT")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "checkpoints", "best_model.pth")


def get_tapvid_pkl_path(dataset_name: str) -> str:
    """Resolve the TAP-Vid dataset file (or kinetics folder of shards).

    Override the root with the TAPVID_ROOT environment variable. Expected layout
    under that root:
        tapvid_davis/tapvid_davis.pkl
        tapvid_kinetics/test/         (folder of .pkl shards)
        tapvid_rgb_stacking/tapvid_rgb_stacking.pkl
    """
    dataset_name = dataset_name.lower()
    root = os.environ.get("TAPVID_ROOT", "./datasets")

    layouts = {
        "davis":        f"{root}/tapvid_davis/tapvid_davis.pkl",
        "kinetics":     f"{root}/tapvid_kinetics/test/",
        "rgb_stacking": f"{root}/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl",
    }
    if dataset_name not in layouts:
        raise ValueError(f"Unknown TAP-Vid dataset '{dataset_name}'. Expected davis/kinetics/rgb_stacking.")
    return standardize_path(layouts[dataset_name])


def crop_custom(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Crop a `size`-pixel square centered at (cx, cy). Out-of-bounds is zero-padded."""
    h, w = image.shape[:2]
    half = size // 2
    cx, cy = int(round(float(cx))), int(round(float(cy)))
    x1, y1 = cx - half, cy - half
    crop = np.zeros((size, size, 3), dtype=image.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x1 + size), min(h, y1 + size)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - x1, sy1 - y1
        dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
        crop[dy1:dy2, dx1:dx2] = image[sy1:sy2, sx1:sx2]
    return crop


def print_official_metrics(dataset_name, OFFICIAL_METRICS, width_or_res_key=None):
    """Pretty-print baseline TAP-Vid metrics from a paper's reported numbers."""
    ds = dataset_name.lower()
    print("\n--- OFFICIAL PAPER METRICS (Baseline) ---")

    if width_or_res_key is not None and width_or_res_key in OFFICIAL_METRICS:
        metrics_dict = OFFICIAL_METRICS[width_or_res_key]
    else:
        metrics_dict = OFFICIAL_METRICS

    if ds not in metrics_dict:
        print(f"  [No official metrics recorded for {ds}]")
        return

    m = metrics_dict[ds]
    epe_str   = f"{m['EPE']}"   if isinstance(m['EPE'],   str) else f"{m['EPE']:.2f}"
    mte_str   = f"{m['MTE']}"   if isinstance(m['MTE'],   str) else f"{m['MTE']:.2f}"
    delta_str = f"{m['delta']}" if isinstance(m['delta'], str) else f"{m['delta']:.2f}%"
    aj_str    = f"{m['AJ']}"    if isinstance(m['AJ'],    str) else f"{m['AJ']:.2f}%"
    oa_str    = f"{m['OA']}"    if isinstance(m['OA'],    str) else f"{m['OA']:.2f}%"
    print(f"  EPE (Mean): {epe_str} px")
    print(f"  MTE (Median): {mte_str} px")
    print(f"  Avg Position Accuracy (delta): {delta_str}")
    print(f"  Average Jaccard (AJ): {aj_str}")
    print(f"  Occlusion Accuracy (OA): {oa_str}")


normalizer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def refine_batch_with_fb(model, templates_128, templates_192, searches_192,
                         scale_factor, num_iters, fb_threshold):
    """Multi-iteration refinement with a forward-backward cycle-consistency check.

    For each forward iteration k=1..N we run k backward iterations from the
    predicted search position. We keep the iteration with the lowest cycle
    error, and fall back to iteration 1 whenever the best cycle error exceeds
    `fb_threshold` (in pixels).

    Returns:
        (deltas_px, num_fb_rejects) — (B, 2) pixel-space deltas and the number
        of points that fell back to iteration 1.
    """
    B = templates_128.shape[0]
    device = templates_128.device

    with torch.no_grad():
        feat_t = model._extract_features(templates_128)

        cx = torch.full((B,), 96.0, device=device)
        cy = torch.full((B,), 96.0, device=device)
        accumulated = torch.zeros((B, 2), device=device)
        snapshots = []  # (acc_delta_norm, cx, cy) per iteration

        for _ in range(num_iters):
            cx = cx.clamp(64.0, 128.0)
            cy = cy.clamp(64.0, 128.0)
            top_left = torch.stack([cx - 64.0, cy - 64.0], dim=1)
            crop = model.extract_patch(searches_192, top_left, 128)
            feat_s = model._extract_features(crop)
            delta = model._fuse_and_regress(feat_t, feat_s)
            accumulated = accumulated + delta[:, :2]
            cx = cx + delta[:, 0] * scale_factor
            cy = cy + delta[:, 1] * scale_factor
            snapshots.append((accumulated.clone(), cx.clone(), cy.clone()))

        best_deltas_norm = snapshots[0][0].clone()
        best_cycle = torch.full((B,), float('inf'), device=device)

        for k, (acc_k, cx_k, cy_k) in enumerate(snapshots):
            tl_k = torch.stack([cx_k.clamp(64.0, 128.0) - 64.0,
                                cy_k.clamp(64.0, 128.0) - 64.0], dim=1)
            feat_bwd = model._extract_features(model.extract_patch(searches_192, tl_k, 128))

            bwd_cx = torch.full((B,), 96.0, device=device)
            bwd_cy = torch.full((B,), 96.0, device=device)
            acc_bwd_px = torch.zeros((B, 2), device=device)

            for _ in range(k + 1):
                bwd_cx = bwd_cx.clamp(64.0, 128.0)
                bwd_cy = bwd_cy.clamp(64.0, 128.0)
                bwd_tl = torch.stack([bwd_cx - 64.0, bwd_cy - 64.0], dim=1)
                d_bwd = model._fuse_and_regress(
                    feat_bwd,
                    model._extract_features(model.extract_patch(templates_192, bwd_tl, 128)))
                acc_bwd_px[:, 0] += d_bwd[:, 0] * scale_factor
                acc_bwd_px[:, 1] += d_bwd[:, 1] * scale_factor
                bwd_cx = bwd_cx + d_bwd[:, 0] * scale_factor
                bwd_cy = bwd_cy + d_bwd[:, 1] * scale_factor

            cycle_err = torch.norm(acc_k * scale_factor + acc_bwd_px, dim=1)
            improved = cycle_err < best_cycle
            best_cycle[improved] = cycle_err[improved]
            best_deltas_norm[improved] = acc_k[improved]

        rejected = best_cycle > fb_threshold
        num_rejects = int(rejected.sum().item())
        if rejected.any():
            best_deltas_norm[rejected] = snapshots[0][0][rejected]

    return best_deltas_norm * scale_factor, num_rejects


def load_refiner(weights_path: str, device: str = None, force_pmr: bool = False):
    """Load a RefinementNetwork from a checkpoint, reading config from config.json.

    Returns (model, scale_factor).
    """
    from subpixr.model import RefinementNetwork

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = os.path.join(os.path.dirname(weights_path), "config.json")
    cfg = json.load(open(cfg_path, "r")) if os.path.exists(cfg_path) else {}

    scale_factor = float(cfg.get("scale_factor", 16.0))
    is_hybrid = float(cfg.get("use_hybrid_fusion", False))
    use_pmr = True if force_pmr else cfg.get("use_pmr_confidence", False)

    model = RefinementNetwork(
        encoder_type=cfg.get("encoder_type", "resnet18"),
        freeze_encoder=cfg.get("freeze_encoder", False),
        dropout_rate=cfg.get("dropout_rate", 0.4),
        use_depthwise_xcorr=cfg.get("use_depthwise_xcorr", True),
        use_attention=cfg.get("use_attention", False),
        predict_confidence=cfg.get("predict_confidence", False),
        use_local_cost_volume=cfg.get("use_local_cost_volume", False),
        scale_factor=scale_factor,
        use_spatial_head=cfg.get("use_spatial_head", False),
        use_multi_stage_features=cfg.get("use_multi_stage_features", False),
        use_pmr_confidence=use_pmr,
        use_hybrid_fusion=is_hybrid,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        msg = str(e)
        if "Missing key(s)" in msg and "encoder.layer" in msg:
            raise RuntimeError(
                f"Checkpoint '{weights_path}' was trained with an older encoder key naming "
                f"scheme incompatible with the current architecture."
            ) from None
        if "size mismatch" in msg and "regressor" in msg:
            raise RuntimeError(
                f"Checkpoint '{weights_path}' has a regressor shape mismatch — check config.json."
            ) from None
        raise

    model.eval()
    return model, scale_factor
