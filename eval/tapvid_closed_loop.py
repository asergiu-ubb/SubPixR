"""
tracker_online_closed_loop.py
online/vs offline

-The Latency Barrier: Offline trackers (like CoTracker and TAPIR in their default modes) require the entire video to be captured before they can output a trajectory.
This introduces a latency equal to the length of the video.
-The Autonomy Bottleneck: In robotics, autonomous driving, and surgical navigation, decisions must be made in milliseconds.
A self-driving car cannot wait for a 10-second video clip to finish recording before deciding if a pedestrian is moving into the crosswalk.
-The "Blind" Causality: Online tracking is strictly causal.
At frame $t$, the system only has access to frames $0$ through $t$.
This forces the tracker to handle occlusions, motion blur, and deformations without the crutch of future context.
Offline trackers (like CoTracker and TAPIR in their default modes) require the entire video to be captured before
they can output a trajectory. This introduces a latency equal to the length of the video.



Unified online + closed-loop refiner evaluation on TAP-Vid datasets.

All three trackers are driven with the same sliding-window schedule:
  WINDOW_LEN = 16 frames  /  STEP = 8 frames

  cotracker  — CoTrackerOnlinePredictor  (true recurrent state across windows)
  tapir      — Causal BootsTAPIR         (stateless, re-anchored each window)
  locotrack  — LocoTrack                 (stateless, re-anchored each window)

For stateless trackers (tapir / locotrack):
  • Each window call is independent — no hidden state persists between windows.
  • Points whose query frame falls within the current window use their GT position.
  • Points already initialised use their last-known prediction as the anchor for
    the first frame of the window.
  • Write policy controls future-information leakage for the "online" claim:
      "first_write" (default): each frame is written by the FIRST window that
        covers it and never overwritten. In steady state this gives at most
        WINDOW_LEN-1 frames of look-ahead on the first window, and STEP-1 frames
        on subsequent windows — a strict upper bound on smoothing.
      "latest_wins": legacy behaviour — every window overwrites, so the final
        prediction at frame t can come from a pass that saw up to ~15 frames
        after t. Not a valid causal/online eval. Kept for reproducing old runs.
      --strict_causal: sets STEP=1 and processes one window per frame ending at
        t, writing only frame t. Zero look-ahead. Slow but honest.

Closed-loop refiner (Phase 2, identical for all trackers):
  • search  at t  = 128px crop centred on tracker's prediction at t
  • template at t = 128px crop at REFINER's own output at t-1   ← closed-loop
  • Template at t0 is seeded from the GT position.

Usage:
  cd eval_sota_with_refiner
  python tracker_online_closed_loop.py davis --tracker cotracker
  python tracker_online_closed_loop.py davis --tracker tapir
  python tracker_online_closed_loop.py davis --tracker locotrack
  python tracker_online_closed_loop.py davis --tracker tapir --refiner_path /path/to/best_model.pth
"""
import sys, os, argparse, pickle, io
from pathlib import Path
from PIL import Image
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import cv2
# ── path setup ───────────────────────────────────────────────────────────────
CO_TRACKER_ROOT  = "./co-tracker"
TAPNET_ROOT      = "./tapnet"
LOCOTRACK_ROOT   = "./locotrack"
LOCOTRACK_PT_ROOT = os.path.join(LOCOTRACK_ROOT, "locotrack_pytorch")

sys.path.insert(0, CO_TRACKER_ROOT)
sys.path.insert(0, TAPNET_ROOT)
sys.path.insert(0, LOCOTRACK_PT_ROOT)
sys.path.append(".")
sys.path.append("..")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from subpixr.utils import get_tapvid_pkl_path, get_general_refiner, load_refiner
from eval.tapvid_metrics import compute_tapvid_metrics

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, default="davis",
                    help="TAP-Vid split: davis | kinetics | rgb_stacking")
parser.add_argument("--tracker", type=str, default="cotracker",
                    choices=["cotracker", "tapir", "locotrack"],
                    help="Which tracker to use as the motion prior")
parser.add_argument("--refiner_path", type=str, default=None,
                    help="Path to best_model.pth. Defaults to get_general_refiner().")
parser.add_argument("--tracker_res", type=int, default=256,
                    help="Internal resolution for tracker (height=width). Default 256.")
parser.add_argument("--cache_dir", type=str, default="./tracker_cache",
                    help="Directory for Phase-1 tracker prediction cache.")
parser.add_argument("--no_cache", action="store_true",
                    help="Disable Phase-1 cache (always re-run the tracker).")
parser.add_argument("--mode", type=str, default="online",
                    choices=["online", "offline"],
                    help="online: sliding-window tracker; "
                         "offline: full-video (more accurate, Mode C).")
parser.add_argument("--cotracker_npz_cache", type=str, default=None,
                    help="Path to cotracker_with_refiner.py's per-sequence .npz cache dir "
                         "(e.g. ./cache_cotracker/davis). Used as fallback when "
                         "--tracker cotracker --mode offline and the primary cache misses.")
parser.add_argument("--locotrack_npz_cache", type=str, default=None,
                    help="Path to locotrack_with_refiner.py's per-sequence .npz cache dir "
                         "(e.g. ./cache_locotrack/davis). Used as fallback when "
                         "--tracker locotrack and the primary cache misses.")
parser.add_argument("--tapir_npz_cache", type=str, default=None,
                    help="Path to tapir_with_refiner.py's per-sequence .npz cache dir "
                         "(e.g. ./cache_tapir/davis). Used as fallback when "
                         "--tracker tapir and the primary cache misses.")
parser.add_argument("--online_write_policy", type=str, default="first_write",
                    choices=["first_write", "latest_wins"],
                    help="How sliding-window trackers (tapir/locotrack) commit "
                         "predictions. 'first_write' is causally honest (no "
                         "re-smoothing from later windows). 'latest_wins' is the "
                         "legacy behaviour with ~15 frames of hidden look-ahead.")
parser.add_argument("--strict_causal", action="store_true",
                    help="Online mode only. Run one window per frame (STEP=1), "
                         "window = [max(0,t-WINDOW_LEN+1), t], write only frame t. "
                         "Strictly zero look-ahead. Slow (T tracker calls).")
parser.add_argument("--sc_threshold", type=float, default=0.0,
                    help="Self-confirming template update threshold (px). "
                         "0.0 (default) = EMA blending (alpha=0.80). "
                         ">0 = update template only when |delta| < threshold, "
                         "matching the SC strategy in cotracker_with_refiner.py. "
                         "Recommended: 1.5")
parser.add_argument("--ema_alpha", type=float, default=0.80,
                    help="EMA blending factor for template update (default 0.80). "
                         "new_template = alpha * old + (1-alpha) * current_crop. "
                         "Used by all EMA-based sections (online/offline/TCL). "
                         "Sweep {0.5, 0.7, 0.8, 0.9, 0.95} to profile closed-loop sensitivity.")
parser.add_argument("--true_closed_loop", action="store_true",
                    help="ONLINE MODE ONLY. Feed the refiner's output back to the "
                         "tracker as the anchor for subsequent windows. "
                         "TAPIR / LocoTrack: replaces base_tracks[win_start] with "
                         "refined_ema[win_start] when seeding each window. "
                         "CoTracker: re-initialises the recurrent state at every "
                         "window with refined queries (destroys cross-window "
                         "recurrent benefit; included for completeness). "
                         "Disables the Phase-1 cache because base_tracks now "
                         "depends on the refiner. Incompatible with --sc_threshold.")
parser.add_argument("--num_iters", type=int,
                    default=int(os.environ.get("TCL_NUM_ITERS", "1")),
                    help="Refiner test-time unroll iterations. 1 = single forward pass "
                         "(default, matches the paper). >1 crops a 192px search canvas and "
                         "unrolls the refiner that many times per frame. Also settable via "
                         "the TCL_NUM_ITERS environment variable.")
parser.add_argument("--no_vis_gate", action="store_true",
                    help="Ablation (R1-W3): also update the EMA template on frames the "
                         "visibility gate would skip (tracker-occluded / GT-occluded), "
                         "exposing template contamination. Default (gate on) only updates "
                         "the template on frames the tracker reports visible.")
args, _ = parser.parse_known_args()

DATASET              = args.dataset
TRACKER              = args.tracker
MODE                 = args.mode
SC_THRESHOLD         = args.sc_threshold
EMA_ALPHA            = args.ema_alpha
ONLINE_WRITE_POLICY  = args.online_write_policy
STRICT_CAUSAL        = args.strict_causal
TRUE_CLOSED_LOOP     = args.true_closed_loop
NUM_ITERS            = args.num_iters
VIS_GATE             = not args.no_vis_gate   # gate the EMA template update on tracker visibility
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_RES   = 256      # TAP-Vid evaluation resolution
QUERY_MODE   = "first"
CROP_SIZE    = 128
# Single-pass uses the 128px search crop; unrolling needs the larger 192px canvas
# so the refiner's internal recurrent re-crop has room to move (RefinementNet.forward).
REFINE_SEARCH = 192 if NUM_ITERS > 1 else CROP_SIZE
WINDOW_LEN   = 16
STEP         = 8
TRACKER_RES  = args.tracker_res   # used for tapir / locotrack

CT_CHECKPOINT         = os.path.join(CO_TRACKER_ROOT, "checkpoints", "scaled_online.pth")
CT_OFFLINE_CHECKPOINT = os.path.join(CO_TRACKER_ROOT, "checkpoints", "scaled_offline.pth")
TAPIR_CHECKPOINT      = os.path.join(TAPNET_ROOT, "checkpoints", "causal_bootstapir_checkpoint.pt")
# Only the causal checkpoint is available; use causal inference for both modes
# (matches tapir_with_refiner.py). Loading causal weights into a non-causal model
# silently produced garbage tracking (~44px EPE baseline on DAVIS).
TAPIR_OFFLINE_CKPT    = TAPIR_CHECKPOINT

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

OFFICIAL = {
    "cotracker": {"davis": {"AJ": 64.4, "OA": 91.2}},
    "tapir":     {"davis": {"AJ": 61.3, "OA": 88.8}},
    "locotrack": {"davis": {"AJ": 63.0, "OA": 87.2}},
}


# ── GPU helpers ───────────────────────────────────────────────────────────────
def frame_to_gpu(frame_bgr: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 BGR → normalised [3,H,W] GPU float (for refiner)."""
    t = (torch.from_numpy(frame_bgr[:, :, ::-1].copy())
         .permute(2, 0, 1).float().to(DEVICE) / 255.0)
    return (t - _MEAN.to(DEVICE)[:, None, None]) / _STD.to(DEVICE)[:, None, None]


import cv2
import numpy as np
import os


def batch_crop_gpu(frame_gpu: torch.Tensor,
                   xs: list, ys: list, size: int) -> torch.Tensor:
    """[N, 3, size, size] crops at (xs[i], ys[i]) via grid_sample."""
    H, W = frame_gpu.shape[1], frame_gpu.shape[2]
    N = len(xs)
    img = frame_gpu.unsqueeze(0).expand(N, -1, -1, -1)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
    ys_t = torch.tensor(ys, dtype=torch.float32, device=DEVICE)
    half = size / 2.0
    offs = torch.arange(size, dtype=torch.float32, device=DEVICE) - half + 0.5
    px = (xs_t[:, None, None] + offs[None, None, :]).expand(N, size, size)
    py = (ys_t[:, None, None] + offs[None, :, None]).expand(N, size, size)
    gx = px / (W - 1) * 2.0 - 1.0
    gy = py / (H - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


# ── tracker loading ───────────────────────────────────────────────────────────
def load_tracker_model(tracker_name: str, mode: str = "online"):
    if tracker_name == "cotracker":
        if mode == "offline":
            from cotracker.predictor import CoTrackerPredictor
            if os.path.exists(CT_OFFLINE_CHECKPOINT):
                # window_len=60 matches cotracker_with_refiner.py (offline full-video
                # context). The default (16) gives slightly worse tracks.
                model = CoTrackerPredictor(checkpoint=CT_OFFLINE_CHECKPOINT,
                                           v2=False, offline=True, window_len=60)
            else:
                print("  scaled_offline.pth not found — downloading from HuggingFace...")
                model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        else:
            from cotracker.predictor import CoTrackerOnlinePredictor
            if os.path.exists(CT_CHECKPOINT):
                model = CoTrackerOnlinePredictor(checkpoint=CT_CHECKPOINT)
            else:
                print("  scaled_online.pth not found — downloading from HuggingFace...")
                model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        return model.to(DEVICE).eval()

    elif tracker_name == "tapir":
        from tapnet.torch import tapir_model as tapir_module
        # Always use causal weights + causal inference — matches tapir_with_refiner.py.
        # Non-causal bootstapir_checkpoint.pt is not shipped here; using causal weights
        # in a non-causal model load cleanly but produce broken tracks.
        ckpt   = TAPIR_CHECKPOINT
        causal = True
        dl_hint = ("wget https://storage.googleapis.com/dm-tapnet/bootstap/"
                   "causal_bootstapir_checkpoint.pt -P tapnet/checkpoints/")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"TAPIR checkpoint not found: {ckpt}\nDownload: {dl_hint}")
        model = tapir_module.TAPIR(pyramid_level=1, use_casual_conv=causal)
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        return model.to(DEVICE).eval()

    elif tracker_name == "locotrack":
        from models.locotrack_model import load_model as loco_load
        model = loco_load(model_size="base")
        return model.to(DEVICE).eval()

    raise ValueError(f"Unknown tracker: {tracker_name}")


# ── per-tracker window inference ──────────────────────────────────────────────
def _tapir_infer_window(model, window_rgb: np.ndarray,
                        queries_win: list, W: int, H: int) -> tuple:
    """
    Run TAPIR on a window of frames.
    queries_win: [[t_rel, y_256, x_256], ...]  (y before x — TAPIR convention)
    Returns: tracks_native [T_win, Nq, 2], vis [T_win, Nq]  (native pixels)
    """
    T_win = len(window_rgb)
    video_in = np.stack([cv2.resize(f, (TRACKER_RES, TRACKER_RES),
                                    cv2.INTER_LINEAR) for f in window_rgb],
                        axis=0).astype(np.float32)
    video_in = (video_in / 255.0) * 2.0 - 1.0
    video_t  = torch.from_numpy(video_in).unsqueeze(0).to(DEVICE)   # [1,T,H,W,3]
    query_t  = torch.tensor([queries_win], dtype=torch.float32,
                             device=DEVICE)                           # [1,N,3]
    with torch.no_grad():
        try:
            out = model(video_t, query_t, query_chunk_size=64)
        except TypeError:
            out = model(video_t, query_t)

    if isinstance(out, dict):
        tr = out["tracks"][0].cpu().numpy()    # [N, T_win, 2] or [T_win, N, 2]
        oc = out["occlusion"][0].cpu().numpy()
    else:
        tr = out[0][0].cpu().numpy()
        oc = out[1][0].cpu().numpy()

    # Normalise to [T_win, N, 2]
    if tr.shape[0] == len(queries_win):        # [N, T_win, 2]
        tr = tr.transpose(1, 0, 2)
    if oc.shape[0] == len(queries_win):        # [N, T_win]
        oc = oc.T

    tr[..., 0] *= W / float(TRACKER_RES)
    tr[..., 1] *= H / float(TRACKER_RES)
    vis = (oc <= 0.0)
    return tr.astype(np.float32), vis.astype(bool)




def _locotrack_infer_window(model, window_rgb: np.ndarray,
                            query_xy_native: np.ndarray,
                            W: int, H: int) -> tuple:
    """
    Run LocoTrack on a window of frames at 384x512 with Aspect Ratio preservation.
    query_xy_native: [Nq, 2]  x, y positions in native pixels (first-frame anchors)
    Returns: tracks_native [T_win, Nq, 2], vis [T_win, Nq]
    """
    H_target, W_target = 384, 512

    # 1. Calculate scaling and padding to preserve aspect ratio
    scale = min(W_target / float(W), H_target / float(H))
    W_new = int(round(W * scale))
    H_new = int(round(H * scale))

    pad_x = (W_target - W_new) / 2.0
    pad_y = (H_target - H_new) / 2.0

    pad_left = int(np.floor(pad_x))
    pad_right = int(np.ceil(pad_x))
    pad_top = int(np.floor(pad_y))
    pad_bottom = int(np.ceil(pad_y))

    # 2. Resize and pad (letterbox) the video window
    video_in = []
    for f in window_rgb:
        resized = cv2.resize(f, (W_new, H_new), interpolation=cv2.INTER_AREA)
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        video_in.append(padded)

    video_in = np.stack(video_in, axis=0).astype(np.float32)
    video_t = ((torch.from_numpy(video_in).to(DEVICE) / 255.0) * 2.0 - 1.0).unsqueeze(0) # [1, T, 384, 512, 3]

    # 3. Adjust native queries to the padded resolution
    q_xy = torch.from_numpy(query_xy_native.astype(np.float32)).to(DEVICE)
    q_xy[:, 0] = q_xy[:, 0] * scale + pad_left
    q_xy[:, 1] = q_xy[:, 1] * scale + pad_top

    # LocoTrack expects [1, N, 3] representing [t, y, x]
    Nq = q_xy.shape[0]
    t_zeros = torch.zeros((Nq, 1), dtype=torch.float32, device=DEVICE)
    q = torch.cat([t_zeros, q_xy.flip(-1)], dim=-1)  # flip x,y to y,x -> [t, y, x]
    q = q.unsqueeze(0)  # [1, N, 3]

    # 4. Inference
    with torch.no_grad():
        feature_grids = model.get_feature_grids(video_t)
        out = model(video_t, q, feature_grids=feature_grids)

    # 5. Extract, un-pad, and un-scale tracks back to native resolution
    tr = out["tracks"][0].permute(1, 0, 2)  # [T_win, N, 2] -> x, y
    tr[..., 0] = (tr[..., 0] - pad_left) / scale
    tr[..., 1] = (tr[..., 1] - pad_top) / scale
    tr = tr.cpu().numpy().astype(np.float32)

    occ = torch.sigmoid(out["occlusion"][0]).permute(1, 0)  # [T_win, N]
    vis = (occ <= 0.5).cpu().numpy().astype(bool)

    return tr, vis


# ── Phase 1: build base_tracks_native / base_vis ─────────────────────────────
def phase1_cotracker(tracker, video_rgb: np.ndarray,
                     queries_list: list, T: int, W: int, H: int) -> tuple:
    """CoTracker online with recurrent state. Returns [T,Nq,2], [T,Nq]."""
    Nq = len(queries_list)
    queries_t = torch.tensor([queries_list], dtype=torch.float32, device=DEVICE)

    video_t   = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).float().to(DEVICE)
    video_256 = F.interpolate(video_t, size=(TARGET_RES, TARGET_RES),
                               mode="bilinear", align_corners=False)  # [T,3,256,256]

    window_frames: list = []
    is_first_step = True
    step = tracker.step          # 8

    for i in range(T):
        if i % step == 0 and i != 0:
            chunk = torch.stack(window_frames[-WINDOW_LEN:], dim=0).unsqueeze(0)
            if is_first_step:
                tracker(chunk, is_first_step=True,
                        queries=queries_t, add_support_grid=True)
                is_first_step = False
            else:
                tracker(chunk, is_first_step=False, add_support_grid=True)
        window_frames.append(video_256[i])

    if is_first_step:
        chunk = torch.stack(window_frames[-WINDOW_LEN:], dim=0).unsqueeze(0)
        tracker(chunk, is_first_step=True, queries=queries_t, add_support_grid=True)
        chunk = torch.stack(window_frames[-WINDOW_LEN:], dim=0).unsqueeze(0)
        tr_w, vi_w = tracker(chunk, is_first_step=False, add_support_grid=True)
    else:
        chunk = torch.stack(window_frames[-WINDOW_LEN:], dim=0).unsqueeze(0)
        tr_w, vi_w = tracker(chunk, is_first_step=False, add_support_grid=True)

    # tr_w: [1, T_acc, Nq, 2] in 256-space
    tracks_256 = tr_w[0, :T].cpu().numpy()          # [T, Nq, 2]
    vis        = vi_w[0, :T].cpu().numpy().astype(bool)

    tracks_native = tracks_256.copy()
    tracks_native[:, :, 0] *= W / TARGET_RES
    tracks_native[:, :, 1] *= H / TARGET_RES
    return tracks_native, vis


def _run_sliding_window(tracker_name: str, tracker,
                        window_rgb: np.ndarray,
                        win_start: int, win_end: int,
                        base_tracks: np.ndarray,
                        queries_list: list, idxs: list,
                        points: np.ndarray, W: int, H: int) -> tuple:
    """Run one tracker window. Returns (tr_win [T_win,Nq,2], vi_win [T_win,Nq])."""
    Nq = len(queries_list)
    if tracker_name == "tapir":
        queries_win = []
        for qn in range(Nq):
            t0 = int(queries_list[qn][0])
            if win_start <= t0 < win_end:
                t_rel = t0 - win_start
                x_q   = float(points[idxs[qn], t0, 0]) * TRACKER_RES
                y_q   = float(points[idxs[qn], t0, 1]) * TRACKER_RES
            else:
                t_rel = 0
                x_q   = base_tracks[win_start, qn, 0] / W * TRACKER_RES
                y_q   = base_tracks[win_start, qn, 1] / H * TRACKER_RES
            queries_win.append([float(t_rel), float(y_q), float(x_q)])
        return _tapir_infer_window(tracker, window_rgb, queries_win, W, H)

    # locotrack
    query_xy = base_tracks[win_start].copy()
    for qn in range(Nq):
        t0 = int(queries_list[qn][0])
        if win_start <= t0 < win_end:
            query_xy[qn, 0] = float(points[idxs[qn], t0, 0]) * W
            query_xy[qn, 1] = float(points[idxs[qn], t0, 1]) * H
    return _locotrack_infer_window(tracker, window_rgb, query_xy, W, H)


def phase1_sliding(tracker_name: str, tracker,
                   video_rgb: np.ndarray, queries_list: list,
                   idxs: list, points: np.ndarray,
                   T: int, W: int, H: int,
                   write_policy: str = "first_write",
                   strict_causal: bool = False) -> tuple:
    """
    Sliding-window inference for TAPIR or LocoTrack (stateless).

    write_policy:
      "first_write"  — each frame written once by the first window covering it.
                       Bounded look-ahead: up to WINDOW_LEN-1 on the very first
                       window, STEP-1 on subsequent ones.
      "latest_wins"  — legacy; every window overwrites. Up to ~15 frames of
                       hidden look-ahead. Not a valid causal eval.

    strict_causal: if True, ignore write_policy and instead run one window per
    frame ending at t, writing only frame t. Zero look-ahead.

    Returns base_tracks_native [T, Nq, 2], base_vis [T, Nq].
    """
    Nq = len(queries_list)

    base_tracks = np.zeros((T, Nq, 2), dtype=np.float32)
    base_vis    = np.zeros((T, Nq), dtype=bool)
    for qn in range(Nq):
        t0, x256, y256 = queries_list[qn]
        base_tracks[:, qn] = [x256 * W / TARGET_RES, y256 * H / TARGET_RES]

    # ── Strict-causal path: one window ending at each frame ──────────────────
    if strict_causal:
        written = np.zeros(T, dtype=bool)
        for t in range(T):
            # Only run if at least one query is alive (t >= its t0) at this frame
            any_alive = any(int(queries_list[qn][0]) <= t for qn in range(Nq))
            if not any_alive:
                continue
            win_start = max(0, t - WINDOW_LEN + 1)
            win_end   = t + 1
            window    = video_rgb[win_start:win_end]
            tr_win, vi_win = _run_sliding_window(
                tracker_name, tracker, window, win_start, win_end,
                base_tracks, queries_list, idxs, points, W, H)
            # Only the last frame of this window (= frame t) gets committed
            rel = win_end - win_start - 1
            base_tracks[t] = tr_win[rel]
            base_vis[t]    = vi_win[rel]
            written[t]     = True
        return base_tracks, base_vis

    # ── Regular sliding path ─────────────────────────────────────────────────
    written = np.zeros(T, dtype=bool)
    for win_end in range(STEP, T + STEP, STEP):
        win_end   = min(win_end, T)
        win_start = max(0, win_end - WINDOW_LEN)
        window    = video_rgb[win_start:win_end]

        tr_win, vi_win = _run_sliding_window(
            tracker_name, tracker, window, win_start, win_end,
            base_tracks, queries_list, idxs, points, W, H)

        T_win = len(window)
        for rel in range(T_win):
            abs_t = win_start + rel
            if not (0 <= abs_t < T):
                continue
            if write_policy == "first_write" and written[abs_t]:
                continue
            base_tracks[abs_t] = tr_win[rel]
            base_vis[abs_t]    = vi_win[rel]
            written[abs_t]     = True

    return base_tracks, base_vis


# ── per-tracker .npz cache compatibility ─────────────────────────────────────

def _load_tracker_npz(npz_dir: str, seq_name: str,
                      W: int, H: int, n_queries: int,
                      vis_from_occ: bool = False) -> tuple | None:
    """
    Try to load a per-sequence .npz written by one of the standalone *_with_refiner.py scripts.
    Returns (base_tracks [T,Nq,2], base_vis [T,Nq]) or None on any mismatch.

    cotracker_with_refiner.py:  keys = tracks, vis, vis_raw, W, H
    locotrack_with_refiner.py:  keys = tracks, vis, W, H
    tapir_with_refiner.py:      keys = tracks, occ, W, H  (occ=occluded → vis = ~occ)
    """
    path = os.path.join(npz_dir, f"{seq_name}.npz")
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=False)
        if int(d["W"]) != W or int(d["H"]) != H:
            return None
        if d["tracks"].shape[1] != n_queries:
            return None
        if vis_from_occ:
            vis = ~d["occ"].astype(bool)
        else:
            vis = d["vis"].astype(bool)
        return d["tracks"].astype(np.float32), vis
    except Exception:
        return None


def _load_cotracker_npz(npz_dir: str, seq_name: str,
                         W: int, H: int, n_queries: int) -> tuple | None:
    return _load_tracker_npz(npz_dir, seq_name, W, H, n_queries, vis_from_occ=False)


# ── Phase 1 (offline): full-video inference ──────────────────────────────────

def phase1_cotracker_offline(tracker, video_rgb: np.ndarray,
                              queries_list: list,
                              T: int, W: int, H: int) -> tuple:
    """CoTrackerPredictor: process full video in one shot. Returns [T,Nq,2],[T,Nq]."""
    queries_t = torch.tensor([queries_list], dtype=torch.float32, device=DEVICE)
    # [1, N, 3] — same [t, x256, y256] format CoTrackerPredictor expects

    video_t = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).float().to(DEVICE)
    # [T, 3, H, W] in [0, 255] — CoTrackerPredictor normalises internally
    video_256 = F.interpolate(video_t, size=(TARGET_RES, TARGET_RES),
                              mode="bilinear", align_corners=False)   # [T, 3, 256, 256]
    video_b = video_256.unsqueeze(0)                                  # [1, T, 3, 256, 256]

    with torch.no_grad():
        pred_tracks, pred_vis = tracker(video_b, queries=queries_t)
    # pred_tracks: [1, T, N, 2] in 256-space; pred_vis: [1, T, N]

    tracks_256 = pred_tracks[0, :T].cpu().numpy()    # [T, N, 2]
    vis        = pred_vis[0, :T].cpu().numpy().astype(bool)

    tracks_native = tracks_256.copy()
    tracks_native[:, :, 0] *= W / TARGET_RES
    tracks_native[:, :, 1] *= H / TARGET_RES
    return tracks_native, vis


def phase1_tapir_offline(model, video_rgb: np.ndarray,
                          queries_list: list,
                          T: int, W: int, H: int) -> tuple:
    """
    Non-causal TAPIR on the full video (single call, no sliding window).
    queries_list: [[t0, x256, y256], ...]  — same as online format.
    Returns [T, Nq, 2], [T, Nq].
    """
    # Convert to TAPIR's [t, y, x] convention; t0 is absolute (window = full video)
    queries_full = [[float(q[0]), float(q[2]), float(q[1])] for q in queries_list]
    tr, vis = _tapir_infer_window(model, video_rgb, queries_full, W, H)
    return tr, vis   # [T, N, 2], [T, N]


def phase1_locotrack_offline(model, video_rgb: np.ndarray,
                              queries_list: list,
                              T: int, W: int, H: int) -> tuple:
    """
    LocoTrack on the full video (single call, no sliding window).
    Uses GT positions at each point's query frame t0 as the anchor.
    Returns [T, Nq, 2], [T, Nq].
    """
    query_xy = np.array(
        [[q[1] * W / TARGET_RES, q[2] * H / TARGET_RES] for q in queries_list],
        dtype=np.float32,
    )  # [N, 2] x, y in native pixels at each point's t0
    tr, vis = _locotrack_infer_window(model, video_rgb, query_xy, W, H)
    return tr, vis   # [T, N, 2], [T, N]


# ── Phase 2: closed-loop sequential refinement ───────────────────────────────
def phase2_closed_loop(base_tracks_native: np.ndarray,
                       base_vis: np.ndarray,
                       video_bgr: np.ndarray,
                       queries_list: list,
                       idxs: list,
                       points: np.ndarray,
                       occluded: np.ndarray,
                       refiner, scale_factor: float,
                       T: int, W: int, H: int,
                       sc_threshold: float = 0.0) -> tuple:
    """
    Closed-loop sequential refinement.
    Always runs EMA template blending (alpha=0.80) independently — no SC interference.
    When sc_threshold > 0, SC runs on top of EMA: its search is re-cropped at EMA's
    refined position, and its output commits EMA+delta_sc only when |delta_sc|<threshold
    (template hard-updated); otherwise SC falls back to EMA's position and its template
    stays frozen.
    Returns (tracks_ema, tracks_sc_or_None, epe_base, epe_ref_ema, epe_ref_sc).
    """
    dual = sc_threshold > 0.0
    Nq = len(queries_list)
    refined_ema = base_tracks_native.copy()
    refined_sc  = base_tracks_native.copy() if dual else None

    # Seed both template dicts from GT at t0.
    templates_ema: dict[int, torch.Tensor] = {}
    templates_sc:  dict[int, torch.Tensor] = {}
    for qn in range(Nq):
        t0   = int(queries_list[qn][0])
        gt_x = float(points[idxs[qn], t0, 0]) * W
        gt_y = float(points[idxs[qn], t0, 1]) * H
        refined_ema[t0, qn] = [gt_x, gt_y]
        if dual:
            refined_sc[t0, qn] = [gt_x, gt_y]
        f0 = frame_to_gpu(video_bgr[t0])
        t_crop = batch_crop_gpu(f0, [gt_x], [gt_y], CROP_SIZE)[0].detach()
        templates_ema[qn] = t_crop
        if dual:
            templates_sc[qn] = t_crop.clone()

    epe_base, epe_ref_ema, epe_ref_sc = [], [], []

    for t in range(T):
        active = [
            qn for qn in range(Nq)
            if (t > int(queries_list[qn][0])
                and not occluded[idxs[qn], t]
                and base_vis[t, qn])
        ]
        # Ablation (--no_vis_gate): also blend the EMA template on frames the
        # visibility gate normally skips (tracker-occluded / GT-occluded), so any
        # template contamination from occluders is exposed. Positions and metrics
        # are unchanged --- only the template state is polluted.
        if not VIS_GATE:
            _act = set(active)
            extra = [qn for qn in range(Nq)
                     if (t > int(queries_list[qn][0]) and qn not in _act)]
            if extra:
                _fg = frame_to_gpu(video_bgr[t])
                _ex_xs = [base_tracks_native[t, qn, 0] for qn in extra]
                _ex_ys = [base_tracks_native[t, qn, 1] for qn in extra]
                _ex_search = batch_crop_gpu(_fg, _ex_xs, _ex_ys, REFINE_SEARCH)
                _ex_tmpl   = torch.stack([templates_ema[qn] for qn in extra], dim=0)
                with torch.no_grad():
                    _ex_d = (refiner(_ex_tmpl, _ex_search, num_iters=NUM_ITERS)[:, :2]
                             .cpu().numpy() * scale_factor)
                _ex_rx = [_ex_xs[bi] + float(_ex_d[bi, 0]) for bi in range(len(extra))]
                _ex_ry = [_ex_ys[bi] + float(_ex_d[bi, 1]) for bi in range(len(extra))]
                _ex_newtmpl = batch_crop_gpu(_fg, _ex_rx, _ex_ry, CROP_SIZE)
                for _bi, qn in enumerate(extra):
                    templates_ema[qn] = (EMA_ALPHA * templates_ema[qn]) + ((1.0 - EMA_ALPHA) * _ex_newtmpl[_bi].detach())
        if not active:
            continue

        frame_gpu = frame_to_gpu(video_bgr[t])
        B = len(active)
        s_xs = [base_tracks_native[t, qn, 0] for qn in active]
        s_ys = [base_tracks_native[t, qn, 1] for qn in active]
        searches  = batch_crop_gpu(frame_gpu, s_xs, s_ys, REFINE_SEARCH)         # [B,3,S,S] S=128|192
        tmpl_ema  = torch.stack([templates_ema[qn] for qn in active], dim=0)     # [B,3,128,128]

        # Step 1: EMA forward pass — runs exactly as if alone, no SC interference.
        with torch.no_grad():
            deltas_ema = (refiner(tmpl_ema, searches, num_iters=NUM_ITERS)[:, :2]
                          .cpu().numpy() * scale_factor)                          # [B, 2]
        r_xs_ema = [s_xs[bi] + float(deltas_ema[bi, 0]) for bi in range(B)]
        r_ys_ema = [s_ys[bi] + float(deltas_ema[bi, 1]) for bi in range(B)]
        new_tmpl_ema = batch_crop_gpu(frame_gpu, r_xs_ema, r_ys_ema, CROP_SIZE)

        # Step 2: SC sits on top of EMA — its search is re-cropped at EMA's refined position.
        if dual:
            tmpl_sc = torch.stack([templates_sc[qn] for qn in active], dim=0)
            searches_sc = batch_crop_gpu(frame_gpu, r_xs_ema, r_ys_ema, REFINE_SEARCH)
            with torch.no_grad():
                deltas_sc = (refiner(tmpl_sc, searches_sc, num_iters=NUM_ITERS)[:, :2]
                             .cpu().numpy() * scale_factor)                        # [B, 2]
            # Confident → nudge further from EMA's position; else stay on EMA.
            r_xs_sc = [r_xs_ema[bi] + float(deltas_sc[bi, 0]) for bi in range(B)]
            r_ys_sc = [r_ys_ema[bi] + float(deltas_sc[bi, 1]) for bi in range(B)]
            new_tmpl_sc = batch_crop_gpu(frame_gpu, r_xs_sc, r_ys_sc, CROP_SIZE)

        for bi, qn in enumerate(active):
            rx_ema, ry_ema = r_xs_ema[bi], r_ys_ema[bi]
            refined_ema[t, qn] = [rx_ema, ry_ema]
            templates_ema[qn] = (EMA_ALPHA * templates_ema[qn]) + ((1.0 - EMA_ALPHA) * new_tmpl_ema[bi].detach())

            if dual:
                delta_mag_sc = float((deltas_sc[bi, 0]**2 + deltas_sc[bi, 1]**2)**0.5)
                if delta_mag_sc < sc_threshold:
                    rx_sc, ry_sc = r_xs_sc[bi], r_ys_sc[bi]
                    templates_sc[qn] = new_tmpl_sc[bi].detach()
                else:
                    # Not confident: fall back to EMA's position; keep SC template frozen.
                    rx_sc, ry_sc = rx_ema, ry_ema
                refined_sc[t, qn] = [rx_sc, ry_sc]

            gt_x = float(points[idxs[qn], t, 0]) * W
            gt_y = float(points[idxs[qn], t, 1]) * H
            epe_base.append(((gt_x - s_xs[bi])**2 + (gt_y - s_ys[bi])**2) ** 0.5)
            epe_ref_ema.append(((gt_x - rx_ema)**2 + (gt_y - ry_ema)**2) ** 0.5)
            if dual:
                epe_ref_sc.append(((gt_x - rx_sc)**2 + (gt_y - ry_sc)**2) ** 0.5)

    return refined_ema, refined_sc, epe_base, epe_ref_ema, epe_ref_sc


# ── True closed-loop: refiner output drives tracker anchor ───────────────────
def _refine_frame_inplace(abs_t: int, base_tracks: np.ndarray, base_vis: np.ndarray,
                          refined_ema: np.ndarray, templates_ema: dict,
                          video_bgr: np.ndarray, queries_list: list, idxs: list,
                          points: np.ndarray, occluded: np.ndarray,
                          refiner, scale_factor: float,
                          T: int, W: int, H: int,
                          epe_base: list, epe_ref_ema: list) -> None:
    """
    Apply refiner to a single frame; commit refined positions, update EMA templates,
    and forward-fill the refined position into refined_ema[abs_t+1:] so the next
    window's win_start anchor reads the latest refined location.
    """
    Nq = len(queries_list)
    active = [
        qn for qn in range(Nq)
        if (abs_t > int(queries_list[qn][0])
            and not occluded[idxs[qn], abs_t]
            and base_vis[abs_t, qn])
    ]
    if not active:
        return
    frame_gpu = frame_to_gpu(video_bgr[abs_t])
    B = len(active)
    s_xs = [base_tracks[abs_t, qn, 0] for qn in active]
    s_ys = [base_tracks[abs_t, qn, 1] for qn in active]
    searches = batch_crop_gpu(frame_gpu, s_xs, s_ys, REFINE_SEARCH)
    tmpl_ema = torch.stack([templates_ema[qn] for qn in active], dim=0)
    with torch.no_grad():
        deltas = (refiner(tmpl_ema, searches, num_iters=NUM_ITERS)[:, :2]
                  .cpu().numpy() * scale_factor)
    r_xs = [s_xs[bi] + float(deltas[bi, 0]) for bi in range(B)]
    r_ys = [s_ys[bi] + float(deltas[bi, 1]) for bi in range(B)]
    new_tmpl = batch_crop_gpu(frame_gpu, r_xs, r_ys, CROP_SIZE)

    for bi, qn in enumerate(active):
        refined_ema[abs_t, qn] = [r_xs[bi], r_ys[bi]]
        # Forward-fill so future windows seeded at win_start > abs_t read this.
        if abs_t + 1 < T:
            refined_ema[abs_t + 1:, qn] = [r_xs[bi], r_ys[bi]]
        templates_ema[qn] = (EMA_ALPHA * templates_ema[qn]) + ((1.0 - EMA_ALPHA) * new_tmpl[bi].detach())

        gt_x = float(points[idxs[qn], abs_t, 0]) * W
        gt_y = float(points[idxs[qn], abs_t, 1]) * H
        epe_base.append(((gt_x - s_xs[bi])**2 + (gt_y - s_ys[bi])**2) ** 0.5)
        epe_ref_ema.append(((gt_x - r_xs[bi])**2 + (gt_y - r_ys[bi])**2) ** 0.5)


def _seed_tcl_buffers(queries_list: list, idxs: list, points: np.ndarray,
                      video_bgr: np.ndarray, T: int, W: int, H: int) -> tuple:
    """Allocate and seed (base_tracks, base_vis, refined_ema, templates_ema)."""
    Nq = len(queries_list)
    base_tracks = np.zeros((T, Nq, 2), dtype=np.float32)
    base_vis    = np.zeros((T, Nq), dtype=bool)
    refined_ema = np.zeros((T, Nq, 2), dtype=np.float32)

    for qn in range(Nq):
        _, x256, y256 = queries_list[qn]
        seed_xy = [x256 * W / TARGET_RES, y256 * H / TARGET_RES]
        base_tracks[:, qn] = seed_xy
        refined_ema[:, qn] = seed_xy

    templates_ema: dict = {}
    for qn in range(Nq):
        t0   = int(queries_list[qn][0])
        gt_x = float(points[idxs[qn], t0, 0]) * W
        gt_y = float(points[idxs[qn], t0, 1]) * H
        # Plant GT at t0 for both buffers; forward-fill so windows starting after
        # t0 but before any refinement step still read a sensible anchor.
        base_tracks[t0:, qn] = [gt_x, gt_y]
        refined_ema[t0:, qn] = [gt_x, gt_y]
        f0 = frame_to_gpu(video_bgr[t0])
        templates_ema[qn] = batch_crop_gpu(f0, [gt_x], [gt_y], CROP_SIZE)[0].detach()
    return base_tracks, base_vis, refined_ema, templates_ema


def true_cl_sliding(tracker_name: str, tracker,
                    video_rgb: np.ndarray, video_bgr: np.ndarray,
                    queries_list: list, idxs: list,
                    points: np.ndarray, occluded: np.ndarray,
                    refiner, scale_factor: float,
                    T: int, W: int, H: int,
                    write_policy: str = "first_write") -> tuple:
    """
    True closed-loop sliding-window inference for TAPIR / LocoTrack.

    For each window, _run_sliding_window reads refined_ema[win_start] (instead
    of the raw tracker output) as the per-point anchor, so the refiner's
    correction at the previous window's last frame propagates into the next
    window's tracker call. After the tracker runs, the refiner is applied to
    every newly-committed frame (per write_policy) and the refined position is
    forward-filled into refined_ema so subsequent windows pick it up.

    Returns base_tracks, base_vis, refined_ema, epe_base, epe_ref_ema.
    """
    base_tracks, base_vis, refined_ema, templates_ema = _seed_tcl_buffers(
        queries_list, idxs, points, video_bgr, T, W, H)

    written = np.zeros(T, dtype=bool)
    epe_base, epe_ref_ema = [], []

    for win_end in range(STEP, T + STEP, STEP):
        win_end   = min(win_end, T)
        win_start = max(0, win_end - WINDOW_LEN)
        window    = video_rgb[win_start:win_end]

        # KEY: anchor for this window's tracker call comes from refined_ema,
        # not raw base_tracks. _run_sliding_window's signature names the
        # parameter base_tracks but only reads index [win_start, qn].
        tr_win, vi_win = _run_sliding_window(
            tracker_name, tracker, window, win_start, win_end,
            refined_ema,
            queries_list, idxs, points, W, H)

        T_win = len(window)
        for rel in range(T_win):
            abs_t = win_start + rel
            if not (0 <= abs_t < T):
                continue
            if write_policy == "first_write" and written[abs_t]:
                continue
            base_tracks[abs_t] = tr_win[rel]
            base_vis[abs_t]    = vi_win[rel]
            written[abs_t]     = True

            _refine_frame_inplace(
                abs_t, base_tracks, base_vis, refined_ema, templates_ema,
                video_bgr, queries_list, idxs, points, occluded,
                refiner, scale_factor, T, W, H,
                epe_base, epe_ref_ema)

    return base_tracks, base_vis, refined_ema, epe_base, epe_ref_ema


def true_cl_cotracker(tracker, video_rgb: np.ndarray, video_bgr: np.ndarray,
                      queries_list: list, idxs: list,
                      points: np.ndarray, occluded: np.ndarray,
                      refiner, scale_factor: float,
                      T: int, W: int, H: int,
                      write_policy: str = "first_write") -> tuple:
    """
    True closed-loop CoTracker (online). The only way to inject refined
    positions into CoTrackerOnlinePredictor is to call it with is_first_step=True
    and fresh queries — which destroys the recurrent state. We therefore reset
    the state at every window, treating CoTracker as a per-window stateless
    tracker. Expected to underperform vanilla recurrent CoTracker; included so
    the comparison is consistent with TAPIR / LocoTrack.

    Returns base_tracks, base_vis, refined_ema, epe_base, epe_ref_ema.
    """
    Nq = len(queries_list)
    base_tracks, base_vis, refined_ema, templates_ema = _seed_tcl_buffers(
        queries_list, idxs, points, video_bgr, T, W, H)

    video_t   = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).float().to(DEVICE)
    video_256 = F.interpolate(video_t, size=(TARGET_RES, TARGET_RES),
                               mode="bilinear", align_corners=False)  # [T,3,256,256]

    written = np.zeros(T, dtype=bool)
    epe_base, epe_ref_ema = [], []

    for win_end in range(STEP, T + STEP, STEP):
        win_end   = min(win_end, T)
        win_start = max(0, win_end - WINDOW_LEN)
        T_win     = win_end - win_start
        chunk     = video_256[win_start:win_end].unsqueeze(0)  # [1, T_win, 3, 256, 256]

        # Build queries: refined position at win_start (or GT at t0 if t0 falls
        # within this window). Coordinates in 256-space, time relative to window.
        queries_win = []
        for qn in range(Nq):
            t0 = int(queries_list[qn][0])
            if win_start <= t0 < win_end:
                t_rel = t0 - win_start
                x256  = float(points[idxs[qn], t0, 0]) * TARGET_RES
                y256  = float(points[idxs[qn], t0, 1]) * TARGET_RES
            else:
                t_rel = 0
                x256  = refined_ema[win_start, qn, 0] * TARGET_RES / W
                y256  = refined_ema[win_start, qn, 1] * TARGET_RES / H
            queries_win.append([float(t_rel), float(x256), float(y256)])
        queries_t = torch.tensor([queries_win], dtype=torch.float32, device=DEVICE)

        # Reset recurrent state with refined queries (destructive).
        with torch.no_grad():
            tracker(chunk, is_first_step=True, queries=queries_t, add_support_grid=True)
            tr_w, vi_w = tracker(chunk, is_first_step=False, add_support_grid=True)
        # tr_w: [1, T_win, Nq, 2] in 256-space
        tracks_256 = tr_w[0, :T_win].cpu().numpy()
        vis_w      = vi_w[0, :T_win].cpu().numpy().astype(bool)
        tr_native  = tracks_256.copy()
        tr_native[:, :, 0] *= W / TARGET_RES
        tr_native[:, :, 1] *= H / TARGET_RES

        for rel in range(T_win):
            abs_t = win_start + rel
            if not (0 <= abs_t < T):
                continue
            if write_policy == "first_write" and written[abs_t]:
                continue
            base_tracks[abs_t] = tr_native[rel]
            base_vis[abs_t]    = vis_w[rel]
            written[abs_t]     = True

            _refine_frame_inplace(
                abs_t, base_tracks, base_vis, refined_ema, templates_ema,
                video_bgr, queries_list, idxs, points, occluded,
                refiner, scale_factor, T, W, H,
                epe_base, epe_ref_ema)

    return base_tracks, base_vis, refined_ema, epe_base, epe_ref_ema


# ── TAP-Vid metric evaluation ─────────────────────────────────────────────────
def eval_tapvid(gt_data: dict, pred_dict: dict) -> dict | None:
    per_seq = []
    for seq_name, seq in gt_data.items():
        if seq_name not in pred_dict:
            continue
        points   = seq["points"]
        occluded = seq["occluded"].astype(bool)
        info     = pred_dict[seq_name]
        pred_px  = info["tracks"]
        pred_vis = info["vis"].astype(bool)
        H, W     = info["H"], info["W"]

        queries, gt_t, gt_o, pr_t, pr_o = [], [], [], [], []
        for i in range(points.shape[0]):
            vis_f = np.where(~occluded[i])[0]
            if len(vis_f) == 0:
                continue
            t0 = int(vis_f[0])
            queries.append([t0,
                             float(points[i, t0, 0]) * TARGET_RES,
                             float(points[i, t0, 1]) * TARGET_RES])
            xy_gt = points[i].astype(np.float32).copy()
            xy_gt[:, 0] *= TARGET_RES
            xy_gt[:, 1] *= TARGET_RES
            gt_t.append(xy_gt)
            gt_o.append(occluded[i])
            xy_pr = pred_px[:, i, :].astype(np.float32).copy()
            xy_pr[:, 0] *= TARGET_RES / float(W)
            xy_pr[:, 1] *= TARGET_RES / float(H)
            pr_t.append(xy_pr)
            pr_o.append(~pred_vis[:, i])

        if not queries:
            continue
        m = compute_tapvid_metrics(
            query_points          = np.array(queries, dtype=np.float32)[None],
            gt_occluded           = np.stack(gt_o, 0)[None],
            gt_tracks             = np.stack(gt_t, 0)[None],
            pred_occluded         = np.stack(pr_o, 0)[None],
            pred_tracks           = np.stack(pr_t, 0)[None],
            query_mode            = QUERY_MODE,
            get_trackwise_metrics = False,
        )
        per_seq.append({k: float(np.array(v).reshape(-1)[0]) for k, v in m.items()})

    if not per_seq:
        return None
    return {k: float(np.mean([d[k] for d in per_seq])) for k in per_seq[0]}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    refiner_path = args.refiner_path or get_general_refiner()

    # Validate true_closed_loop combinations
    if TRUE_CLOSED_LOOP:
        if MODE != "online":
            raise SystemExit("--true_closed_loop only valid with --mode online")
        if SC_THRESHOLD > 0.0:
            raise SystemExit("--true_closed_loop is incompatible with --sc_threshold; "
                             "drop one of them.")
        if STRICT_CAUSAL:
            raise SystemExit("--true_closed_loop is incompatible with --strict_causal "
                             "(would require T tracker calls per sequence); drop one.")

    if MODE == "online":
        if STRICT_CAUSAL and TRACKER in ("tapir", "locotrack"):
            _tracker_detail = f"(window={WINDOW_LEN}, strict_causal, zero look-ahead)"
        elif TRACKER in ("tapir", "locotrack"):
            _tracker_detail = f"(window={WINDOW_LEN}, step={STEP}, policy={ONLINE_WRITE_POLICY})"
        else:
            _tracker_detail = f"(window={WINDOW_LEN}, step={STEP})"
    else:
        _tracker_detail = "(full-video offline)"
    _tmpl_mode = (f"SC thr={SC_THRESHOLD}px" if SC_THRESHOLD > 0.0 else f"EMA alpha={EMA_ALPHA}")
    _loop_mode = ("TRUE closed-loop (refiner→tracker feedback)" if TRUE_CLOSED_LOOP
                  else "open-loop (tracker independent of refiner)")
    print(f"Dataset    : {DATASET}")
    print(f"Tracker    : {TRACKER}  [{MODE}]  {_tracker_detail}")
    print(f"Loop       : {_loop_mode}")
    print(f"Template   : {_tmpl_mode}")
    print(f"Refiner    : {refiner_path}")
    print(f"Device     : {DEVICE}")

    # ── Phase-1 cache ─────────────────────────────────────────────────────────
    # Disable cache in true closed-loop mode: base_tracks now depends on the
    # refiner, so a cached open-loop base_tracks is the wrong starting point.
    use_cache  = (not args.no_cache) and (not TRUE_CLOSED_LOOP)
    if TRUE_CLOSED_LOOP and not args.no_cache:
        print("Note: --true_closed_loop disables the Phase-1 cache "
              "(base_tracks depends on refiner output).")
    # Cache variants for the different online write policies must not collide —
    # they produce different Phase-1 tracks.
    if MODE == "online":
        if STRICT_CAUSAL and TRACKER in ("tapir", "locotrack"):
            _cache_suffix = f"_w{WINDOW_LEN}_s1_strictcausal"
        elif TRACKER in ("tapir", "locotrack"):
            _cache_suffix = f"_w{WINDOW_LEN}_s{STEP}_{ONLINE_WRITE_POLICY}"
        else:
            _cache_suffix = f"_w{WINDOW_LEN}_s{STEP}"
    else:
        _cache_suffix = ""
    cache_path = Path(args.cache_dir) / (
        f"{TRACKER}_{DATASET}_{MODE}_res{TRACKER_RES}{_cache_suffix}.pkl"
    )
    phase1_cache: dict = {}
    if use_cache and cache_path.exists():
        print(f"Loading Phase-1 cache: {cache_path}")
        with open(cache_path, "rb") as _f:
            phase1_cache = pickle.load(_f)
        print(f"  {len(phase1_cache)} sequences cached")

    print("Loading refiner...")
    refiner, scale_factor = load_refiner(refiner_path, DEVICE)
    refiner.eval()

    # Only load the tracker model if there will be cache misses
    tracker = None

    pkl_path = get_tapvid_pkl_path(DATASET)
    print(f"Loading dataset: {pkl_path}")
    if os.path.isdir(pkl_path):
        # Sharded dataset (e.g. Kinetics): load all *.pkl shards and merge.
        # Keep frames as raw JPEG bytes; decode lazily per-sequence in the main
        # loop to avoid loading GBs of decoded pixels into RAM all at once.
        shard_files = sorted(Path(pkl_path).glob("*.pkl"))
        if not shard_files:
            raise FileNotFoundError(f"No .pkl shards found in {pkl_path}")
        data = {}
        for shard in shard_files:
            with open(shard, "rb") as f:
                shard_data = pickle.load(f)
            if isinstance(shard_data, dict):
                shard_data = list(shard_data.values())
            for s in shard_data:
                key = f"seq_{len(data):05d}"
                data[key] = s
            print(f"  shard {shard.name}: {len(shard_data)} seqs  (total so far: {len(data)})")
    else:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            data = {f"seq_{i:04d}": s for i, s in enumerate(data)}
    print(f"  {len(data)} sequences total")

    # Determine how many sequences will need the tracker (load lazily).
    # For cotracker+offline, also count .npz hits as covered.
    cache_misses = [s for s in data if s not in phase1_cache] if use_cache else list(data)
    _npz_cache_arg = {
        "cotracker": args.cotracker_npz_cache,
        "locotrack": args.locotrack_npz_cache,
        "tapir":     args.tapir_npz_cache,
    }.get(TRACKER)
    _npz_vis_from_occ = (TRACKER == "tapir")

    if _npz_cache_arg and not TRUE_CLOSED_LOOP:
        # Pre-check which npz files exist so we don't load the model unnecessarily.
        # Skipped under --true_closed_loop because the tracker must run live.
        _npz_covered = set()
        for s in list(cache_misses):
            seq = data[s]
            _H, _W = seq["video"].shape[1], seq["video"].shape[2]
            _nq = sum(1 for i in range(seq["points"].shape[0])
                      if np.any(~seq["occluded"][i]))
            if _load_tracker_npz(_npz_cache_arg, s, _W, _H, _nq,
                                  vis_from_occ=_npz_vis_from_occ) is not None:
                _npz_covered.add(s)
        cache_misses = [s for s in cache_misses if s not in _npz_covered]
        if _npz_covered:
            print(f"  {len(_npz_covered)} sequences covered by .npz cache "
                  f"({_npz_cache_arg})")
    if cache_misses:
        print(f"Loading {TRACKER} [{MODE}] (needed for {len(cache_misses)} cache misses)...")
        tracker = load_tracker_model(TRACKER, mode=MODE)
    else:
        print(f"All {len(data)} sequences found in cache — skipping tracker load.")

    dual_strategy = SC_THRESHOLD > 0.0

    cache_updated = False
    baseline_results:    dict = {}
    refined_ema_results: dict = {}
    refined_sc_results:  dict = {} if dual_strategy else None
    epe_base_all, epe_ref_ema_all, epe_ref_sc_all = [], [], []

    for seq_idx, (seq_name, seq) in enumerate(data.items()):
        print(f"\r[{seq_idx+1}/{len(data)}] {seq_name:<30}", end="", flush=True)

        # Kinetics stores frames as JPEG bytes; decode lazily (one seq at a time).
        # Resize to TARGET_RES on CPU so the full-res tensor never hits VRAM.
        _raw = seq["video"]
        if len(_raw) > 0 and isinstance(_raw[0], bytes):
            _raw = np.stack([
                cv2.resize(
                    np.array(Image.open(io.BytesIO(f)).convert("RGB")),
                    (TARGET_RES, TARGET_RES), interpolation=cv2.INTER_AREA,
                )
                for f in _raw
            ])
        video_rgb = np.array(_raw)                    # [T,H,W,3] uint8 RGB
        video_bgr = video_rgb[..., ::-1].copy()       # [T,H,W,3] uint8 BGR  (refiner)
        points    = seq["points"]                     # [N, T, 2]  in [0,1]
        occluded  = seq["occluded"]                   # [N, T]
        T, H, W   = video_rgb.shape[:3]

        # ── Build query list ──────────────────────────────────────────────────
        queries_list: list = []   # [[t0, x256, y256], ...]  (CoTracker / TAPIR x,y)
        idxs: list[int]    = []
        t0s:  list[int]    = []
        for i in range(points.shape[0]):
            vis = np.where(~occluded[i])[0]
            if len(vis) == 0:
                continue
            t0 = int(vis[0])
            queries_list.append([
                float(t0),
                float(points[i, t0, 0]) * TARGET_RES,   # x * 256
                float(points[i, t0, 1]) * TARGET_RES,   # y * 256
            ])
            idxs.append(i)
            t0s.append(t0)

        if not queries_list:
            continue

        # ── Phase 1 + Phase 2 ────────────────────────────────────────────────
        if TRUE_CLOSED_LOOP:
            # Interleaved: refiner output drives tracker anchor at each window.
            if TRACKER == "cotracker":
                base_tracks, base_vis, ema_tracks, epe_b, epe_r_ema = true_cl_cotracker(
                    tracker, video_rgb, video_bgr,
                    queries_list, idxs, points, occluded,
                    refiner, scale_factor, T, W, H,
                    write_policy=ONLINE_WRITE_POLICY)
            else:
                base_tracks, base_vis, ema_tracks, epe_b, epe_r_ema = true_cl_sliding(
                    TRACKER, tracker, video_rgb, video_bgr,
                    queries_list, idxs, points, occluded,
                    refiner, scale_factor, T, W, H,
                    write_policy=ONLINE_WRITE_POLICY)
            sc_tracks, epe_r_sc = None, []
        else:
            # Open-loop: Phase 1 (tracker) is independent of the refiner.
            if use_cache and seq_name in phase1_cache:
                cached = phase1_cache[seq_name]
                base_tracks = cached["base_tracks"]
                base_vis    = cached["base_vis"]
            else:
                # Fallback: try the per-sequence .npz written by the standalone
                # *_with_refiner.py scripts (cotracker / locotrack / tapir).
                _npz_hit = False
                if _npz_cache_arg:
                    _npz = _load_tracker_npz(
                        _npz_cache_arg, seq_name, W, H, len(queries_list),
                        vis_from_occ=_npz_vis_from_occ)
                    if _npz is not None:
                        base_tracks, base_vis = _npz
                        _npz_hit = True

                if not _npz_hit:
                    if MODE == "offline":
                        if TRACKER == "cotracker":
                            base_tracks, base_vis = phase1_cotracker_offline(
                                tracker, video_rgb, queries_list, T, W, H)
                        elif TRACKER == "tapir":
                            base_tracks, base_vis = phase1_tapir_offline(
                                tracker, video_rgb, queries_list, T, W, H)
                        else:  # locotrack
                            base_tracks, base_vis = phase1_locotrack_offline(
                                tracker, video_rgb, queries_list, T, W, H)
                    else:
                        if TRACKER == "cotracker":
                            base_tracks, base_vis = phase1_cotracker(
                                tracker, video_rgb, queries_list, T, W, H)
                        else:
                            base_tracks, base_vis = phase1_sliding(
                                TRACKER, tracker, video_rgb, queries_list,
                                idxs, points, T, W, H,
                                write_policy=ONLINE_WRITE_POLICY,
                                strict_causal=STRICT_CAUSAL)

                # Populate the primary .pkl cache (even when loaded from .npz)
                if use_cache:
                    phase1_cache[seq_name] = {
                        "base_tracks": base_tracks,
                        "base_vis":    base_vis,
                    }
                    cache_updated = True

            # Phase 2: closed-loop refiner (EMA template, no tracker feedback)
            ema_tracks, sc_tracks, epe_b, epe_r_ema, epe_r_sc = phase2_closed_loop(
                base_tracks, base_vis, video_bgr,
                queries_list, idxs, points, occluded,
                refiner, scale_factor, T, W, H,
                sc_threshold=SC_THRESHOLD)

        epe_base_all.extend(epe_b)
        epe_ref_ema_all.extend(epe_r_ema)
        if dual_strategy:
            epe_ref_sc_all.extend(epe_r_sc)

        # ── Pack into full-size arrays for TAP-Vid eval ───────────────────────
        N_pts         = points.shape[0]
        ct_full       = np.zeros((T, N_pts, 2), dtype=np.float32)
        ref_ema_full  = np.zeros((T, N_pts, 2), dtype=np.float32)
        ref_sc_full   = np.zeros((T, N_pts, 2), dtype=np.float32) if dual_strategy else None
        vis_full      = np.zeros((T, N_pts), dtype=bool)

        for qn, orig_idx in enumerate(idxs):
            ct_full[:, orig_idx]       = base_tracks[:, qn]
            ref_ema_full[:, orig_idx]  = ema_tracks[:, qn]
            if dual_strategy:
                ref_sc_full[:, orig_idx] = sc_tracks[:, qn]
            vis_full[:, orig_idx] = base_vis[:, qn]
            vis_full[t0s[qn], orig_idx] = True   # query frame always visible

        baseline_results[seq_name]    = {"tracks": ct_full,      "vis": vis_full, "H": H, "W": W}
        refined_ema_results[seq_name] = {"tracks": ref_ema_full, "vis": vis_full, "H": H, "W": W}
        if dual_strategy:
            refined_sc_results[seq_name] = {"tracks": ref_sc_full, "vis": vis_full, "H": H, "W": W}
    # ── Persist cache if updated ──────────────────────────────────────────────
    if use_cache and cache_updated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as _f:
            pickle.dump(phase1_cache, _f)
        print(f"\nPhase-1 cache saved: {cache_path}  ({len(phase1_cache)} sequences)")

    print(f"\n\nComputing TAP-Vid metrics ({len(baseline_results)} sequences)...")
    base_m    = eval_tapvid(data, baseline_results)
    ref_ema_m = eval_tapvid(data, refined_ema_results)
    ref_sc_m  = eval_tapvid(data, refined_sc_results) if dual_strategy else None

    # ── Report ────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    def _pct(m, k):
        return f"{m[k]*100:.2f}%" if m and k in m else "N/A"

    def _epe(arr):
        return f"{np.mean(arr):.4f} px" if arr else "N/A"

    _loop_tag       = " [TCL]" if TRUE_CLOSED_LOOP else ""
    tracker_label   = f"{TRACKER.upper()} {MODE}{_loop_tag}"
    ema_label       = f"{TRACKER.upper()} +EMA{_loop_tag}"
    sc_label        = f"+SC{SC_THRESHOLD}px"
    _mode_detail = (f"  window={WINDOW_LEN}, step={STEP}, tracker_res={TRACKER_RES}"
                    if MODE == "online" else
                    f"  tracker_res={TRACKER_RES}  (full-video offline)")

    if dual_strategy:
        SEP = "─" * 82
        W_L, W_R1, W_R2, W_R3 = 28, 16, 16, 16
        header_title = f"{MODE.capitalize()} + CL-Refiner [EMA + SC{SC_THRESHOLD}px]  —  {TRACKER.upper()}  —  {DATASET.upper()}"
        lines = [
            "=" * 82,
            header_title,
            _mode_detail,
            "=" * 82,
            f"Time     : {ts}",
            f"Refiner  : {refiner_path}",
            f"Pairs    : {len(epe_base_all)}  (GT-visible & tracker-visible)",
            "",
            SEP,
            f"{'Metric':<{W_L}} {tracker_label:>{W_R1}} {ema_label:>{W_R2}} {sc_label:>{W_R3}}",
            SEP,
            f"{'EPE (mean px)':<{W_L}} {_epe(epe_base_all):>{W_R1}} {_epe(epe_ref_ema_all):>{W_R2}} {_epe(epe_ref_sc_all):>{W_R3}}",
            f"{'Avg Position Accuracy':<{W_L}} {_pct(base_m,'average_pts_within_thresh'):>{W_R1}} {_pct(ref_ema_m,'average_pts_within_thresh'):>{W_R2}} {_pct(ref_sc_m,'average_pts_within_thresh'):>{W_R3}}",
            f"{'Average Jaccard (AJ)':<{W_L}} {_pct(base_m,'average_jaccard'):>{W_R1}} {_pct(ref_ema_m,'average_jaccard'):>{W_R2}} {_pct(ref_sc_m,'average_jaccard'):>{W_R3}}",
            f"{'Occlusion Accuracy (OA)':<{W_L}} {_pct(base_m,'occlusion_accuracy'):>{W_R1}} {_pct(ref_ema_m,'occlusion_accuracy'):>{W_R2}} {_pct(ref_sc_m,'occlusion_accuracy'):>{W_R3}}",
            SEP,
        ]
    else:
        SEP = "─" * 68
        W_L, W_R1, W_R2 = 32, 18, 18
        header_title = f"{MODE.capitalize()} + Closed-Loop Refiner [EMA]  —  {TRACKER.upper()}  —  {DATASET.upper()}"
        lines = [
            "=" * 68,
            header_title,
            _mode_detail,
            "=" * 68,
            f"Time     : {ts}",
            f"Refiner  : {refiner_path}",
            f"Pairs    : {len(epe_base_all)}  (GT-visible & tracker-visible)",
            "",
            SEP,
            f"{'Metric':<{W_L}} {tracker_label:>{W_R1}} {ema_label:>{W_R2}}",
            SEP,
            f"{'EPE (mean px)':<{W_L}} {_epe(epe_base_all):>{W_R1}} {_epe(epe_ref_ema_all):>{W_R2}}",
            f"{'Avg Position Accuracy':<{W_L}} {_pct(base_m,'average_pts_within_thresh'):>{W_R1}} {_pct(ref_ema_m,'average_pts_within_thresh'):>{W_R2}}",
            f"{'Average Jaccard (AJ)':<{W_L}} {_pct(base_m,'average_jaccard'):>{W_R1}} {_pct(ref_ema_m,'average_jaccard'):>{W_R2}}",
            f"{'Occlusion Accuracy (OA)':<{W_L}} {_pct(base_m,'occlusion_accuracy'):>{W_R1}} {_pct(ref_ema_m,'occlusion_accuracy'):>{W_R2}}",
            SEP,
        ]

    off = OFFICIAL.get(TRACKER, {}).get(DATASET)
    if off:
        lines.append(f"\nPublished {TRACKER} (offline): AJ={off.get('AJ','?')}%  OA={off.get('OA','?')}%")
        if MODE == "online":
            lines.append(f"(online mode is expected to score lower than offline)")

    lines.append(f"\nFINISHED: tracker_online_closed_loop.py  [{TRACKER}] [{MODE}]")
    report = "\n".join(lines)
    print(report)

    out_dir  = os.path.dirname(refiner_path)
    _iter_tag  = f"_it{NUM_ITERS}" if NUM_ITERS > 1 else ""
    _gate_tag  = "" if VIS_GATE else "_novisgate"
    _sc_tag    = f"_sc{SC_THRESHOLD}" if dual_strategy else ""
    _tcl_tag   = "_tcl" if TRUE_CLOSED_LOOP else ""
    _alpha_tag = f"_ema{EMA_ALPHA}" if EMA_ALPHA != 0.80 else ""
    log_path = os.path.join(out_dir, f"REPORT_{TRACKER}_{MODE}_cl{_iter_tag}{_gate_tag}{_tcl_tag}{_sc_tag}{_alpha_tag}_{DATASET}.txt")
    with open(log_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {log_path}")


if __name__ == "__main__":
    main()
