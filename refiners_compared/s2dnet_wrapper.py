"""
S2DNet hierarchical sparse-to-dense refinement (shared implementation).

Used by:
  - eval_S2DNet_sota.py              (MegaDepth/ScanNet, over matcher H5 preds)
  - hpatches/hpatches_refiner_MMA.py (HPatches Strategy B)

Mirrors the published S2DNet pipeline:
  1. Per-image L2-normalised hyperfeature pyramid (fine → coarse).
  2. Hierarchical coarse-to-fine cosine-similarity search, seeded only from
     the query kp0 (no kp1 prior used as a search window).
  3. Cycle-consistency filter at the finest level.

`refine_hierarchical` returns the per-point delta relative to kp1 plus a
cycle-ok mask; callers decide whether to magnitude-gate or cycle-gate.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Image-pixel stride per feature cell, for every VGG16 layer name S2DNet may
# expose. Within block N, conv/relu layers share stride 2^(N-1); poolN halves
# res going into block N+1 so has stride 2^N.
_VGG16_STRIDES = {
    "conv1_1": 1, "relu1_1": 1, "conv1_2": 1, "relu1_2": 1, "pool1": 2,
    "conv2_1": 2, "relu2_1": 2, "conv2_2": 2, "relu2_2": 2, "pool2": 4,
    "conv3_1": 4, "relu3_1": 4, "conv3_2": 4, "relu3_2": 4,
    "conv3_3": 4, "relu3_3": 4, "pool3": 8,
    "conv4_1": 8, "relu4_1": 8, "conv4_2": 8, "relu4_2": 8,
    "conv4_3": 8, "relu4_3": 8, "pool4": 16,
    "conv5_1": 16, "relu5_1": 16, "conv5_2": 16, "relu5_2": 16,
    "conv5_3": 16, "relu5_3": 16, "pool5": 32,
}

_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)


def load_s2dnet(checkpoint_path: str, device: str):
    """Load the published S2DNet-Minimal checkpoint.

    Returns (model, level_scales) where level_scales is a fine→coarse tuple
    of image-pixel strides matching whatever hypercolumn layers the checkpoint
    exposes (the checkpoint overwrites `_hypercolumn_layers` at load time, so
    this must be derived after load).
    """
    s2d_root = os.path.dirname(os.path.abspath(checkpoint_path))
    _saved = sys.path[:]
    sys.path.insert(0, s2d_root)
    try:
        from s2dnet import S2DNet
    finally:
        sys.path = _saved

    model = S2DNet(device=device, checkpoint_path=checkpoint_path).eval().to(device)

    # S2DNet-Minimal truncates VGG16 at conv5_3 (index 28), but the checkpoint
    # may reference 'relu5_3' (index 29). Append the missing ReLU so the
    # forward loop can reach it.
    if 'relu5_3' in model._hypercolumn_layers and len(list(model.encoder.children())) < 30:
        model.encoder.add_module('29', nn.ReLU(inplace=True))

    level_scales = tuple(float(_VGG16_STRIDES[l]) for l in model._hypercolumn_layers)
    return model, level_scales


def extract_pyramid(img_rgb: np.ndarray, model, device: str) -> list:
    """Extract the L2-normalised hyperfeature pyramid from an RGB image.

    Returns a list of [1, C, H_l, W_l] tensors on `device`, ordered
    FINE → COARSE (matches S2DNet's forward output order, which follows
    VGG depth and the checkpoint's hypercolumn_layers list).
    """
    mean = torch.tensor(_NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_NORM_STD, device=device).view(1, 3, 1, 1)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    t = (t - mean) / std
    with torch.no_grad():
        feat_maps = model(t)
    return [F.normalize(f, dim=1) for f in feat_maps]


def _sample_descriptors(feat: torch.Tensor, xy_img: torch.Tensor, scale: float) -> torch.Tensor:
    """Bilinear-sample descriptors at sub-pixel image coords from a feature map.

    feat   : [1, C, H, W] L2-normalised feature map at stride `scale`.
    xy_img : [N, 2] pixel coords in the IMAGE frame (not feature-map cells).
    Returns [N, C] L2-normalised descriptors (re-normalised after blend).
    """
    _, C, H, W = feat.shape
    fx = (xy_img[:, 0] / scale).clamp(0.0, float(W - 1))
    fy = (xy_img[:, 1] / scale).clamp(0.0, float(H - 1))
    grid = torch.stack([
        2.0 * fx / max(W - 1, 1) - 1.0,
        2.0 * fy / max(H - 1, 1) - 1.0,
    ], dim=-1).view(1, -1, 1, 2).to(feat.dtype)
    desc = F.grid_sample(feat, grid, mode='bilinear',
                         padding_mode='border', align_corners=True)
    desc = desc.squeeze(-1).squeeze(0).t().contiguous()  # [N, C]
    return F.normalize(desc, dim=1)


def _soft_argmax_2d(sim: torch.Tensor, by: int, bx: int,
                    local_r: int = 2, temp: float = 0.1):
    """Sub-pixel localisation around (bx, by) via (2r+1)x(2r+1) softmax CoM."""
    ya = max(0, by - local_r); yb = min(sim.shape[0], by + local_r + 1)
    xa = max(0, bx - local_r); xb = min(sim.shape[1], bx + local_r + 1)
    local = sim[ya:yb, xa:xb]
    prob = F.softmax(local.reshape(-1) / temp, dim=0).view_as(local)
    ygrid, xgrid = torch.meshgrid(
        torch.arange(ya, yb, device=sim.device, dtype=sim.dtype),
        torch.arange(xa, xb, device=sim.device, dtype=sim.dtype),
        indexing='ij',
    )
    return (xgrid * prob).sum().item(), (ygrid * prob).sum().item()


def _search_level(query: torch.Tensor, feat: torch.Tensor,
                  center_xy_img, win_radius_px, scale: float, soft_argmax: bool):
    """Search a single L2-normalised [C] query in a [1,C,H,W] map.

    win_radius_px=None → global search (used at the coarsest level for the
    kp1-free initial guess).
    """
    _, C, H, W = feat.shape
    if win_radius_px is None:
        x1, y1, x2, y2 = 0, 0, W, H
    else:
        rcell = max(1, int(round(win_radius_px / scale)))
        cx_cell = int(round(float(center_xy_img[0]) / scale))
        cy_cell = int(round(float(center_xy_img[1]) / scale))
        x1 = max(0, cx_cell - rcell); y1 = max(0, cy_cell - rcell)
        x2 = min(W, cx_cell + rcell + 1); y2 = min(H, cy_cell + rcell + 1)
        if x2 <= x1 or y2 <= y1:
            return float(center_xy_img[0]), float(center_xy_img[1]), -1.0

    win = feat[0, :, y1:y2, x1:x2].reshape(C, -1)
    sim = (query.unsqueeze(0) @ win).view(y2 - y1, x2 - x1)
    best = int(sim.argmax())
    by = best // sim.shape[1]; bx = best % sim.shape[1]
    peak = float(sim[by, bx].item())
    if soft_argmax:
        sub_x, sub_y = _soft_argmax_2d(sim, by, bx)
    else:
        sub_x, sub_y = float(bx), float(by)
    return (x1 + sub_x) * scale, (y1 + sub_y) * scale, peak


def refine_hierarchical(pyr0: list, pyr1: list, kp0: np.ndarray, kp1: np.ndarray,
                        level_scales: tuple,
                        mid_radius_px: float = 32.0,
                        fine_radius_px: float = 4.0,
                        cycle_thresh: float = 4.0):
    """Coarse-to-fine sparse-to-dense refinement, pyramid-depth agnostic.

    pyr0, pyr1    : L2-normalised feature maps, FINE → COARSE (list).
    kp0           : [N, 2] query kp in image0 (sole search seed).
    kp1           : [N, 2] baseline kp in image1 (only used to compute the
                    returned delta; NOT used as a search prior).
    level_scales  : image-pixel stride per pyramid level (fine→coarse tuple),
                    same length as pyr0/pyr1.

    Strategy:
      - Coarsest level : global (kp1-free) argmax.
      - Intermediate   : hard argmax ±mid_radius_px.
      - Finest level   : soft argmax ±fine_radius_px (sub-pixel CoM).
      - Cycle filter   : round-trip search at the finest level.

    Returns (deltas [N, 2] = refined - kp1, cycle_ok [N] bool mask).
    """
    L = len(pyr0)
    assert L == len(pyr1) == len(level_scales), (
        f"Pyramid/scale mismatch: pyr0={L} pyr1={len(pyr1)} "
        f"scales={len(level_scales)}"
    )
    feat0_fine, feat1_fine = pyr0[0], pyr1[0]
    s_fine = level_scales[0]

    N = len(kp0)
    deltas = np.zeros((N, 2), dtype=np.float64)
    cycle_ok = np.zeros(N, dtype=bool)
    if N == 0:
        return deltas, cycle_ok

    kp0_t = torch.from_numpy(np.asarray(kp0, dtype=np.float32)).to(feat0_fine.device)

    # Sub-pixel queries at every pyramid level for every kp0. [L][N, C]
    queries = [_sample_descriptors(pyr0[l], kp0_t, level_scales[l])
               for l in range(L)]

    # --- Batched COARSE search over the entire coarsest map of image1 ---
    coarse_feat = pyr1[L - 1]
    s_coarse = level_scales[L - 1]
    _, Cc, Hc, Wc = coarse_feat.shape
    flat = coarse_feat[0].reshape(Cc, -1)              # [C, Hc*Wc]
    sim_coarse = (queries[L - 1] @ flat)               # [N, Hc*Wc]
    coarse_idx = sim_coarse.argmax(dim=1)
    coarse_y_cell = (coarse_idx // Wc).float()
    coarse_x_cell = (coarse_idx % Wc).float()
    coarse_x_img = (coarse_x_cell * s_coarse).cpu().numpy()
    coarse_y_img = (coarse_y_cell * s_coarse).cpu().numpy()

    intermediate_levels = list(range(L - 2, 0, -1))

    for i in range(N):
        x, y = float(coarse_x_img[i]), float(coarse_y_img[i])

        for lvl in intermediate_levels:
            x, y, _ = _search_level(
                queries[lvl][i], pyr1[lvl], (x, y),
                win_radius_px=mid_radius_px, scale=level_scales[lvl],
                soft_argmax=False,
            )

        fine_x, fine_y, peak = _search_level(
            queries[0][i], feat1_fine, (x, y),
            win_radius_px=fine_radius_px, scale=s_fine, soft_argmax=True,
        )

        rev_xy = torch.tensor([[fine_x, fine_y]], device=feat1_fine.device, dtype=torch.float32)
        rev_q = _sample_descriptors(feat1_fine, rev_xy, s_fine)[0]
        back_x, back_y, _ = _search_level(
            rev_q, feat0_fine,
            (float(kp0_t[i, 0]), float(kp0_t[i, 1])),
            win_radius_px=max(8.0, fine_radius_px), scale=s_fine, soft_argmax=True,
        )
        round_trip = float(np.hypot(back_x - float(kp0_t[i, 0]),
                                    back_y - float(kp0_t[i, 1])))
        cycle_ok[i] = (round_trip <= cycle_thresh) and (peak > 0.0)

        deltas[i, 0] = fine_x - kp1[i, 0]
        deltas[i, 1] = fine_y - kp1[i, 1]

    return deltas, cycle_ok
