import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Image degradation helpers (applied independently per image during training
# to simulate real-world domain gap: compression, motion, sensor noise).
# ---------------------------------------------------------------------------

def _jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """Simulate JPEG compression artifacts. Input/output: uint8 RGB."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _motion_blur(img: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
    """Apply directional motion blur. Input/output: uint8 RGB."""
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0 / kernel_size
    center = ((kernel_size - 1) / 2, (kernel_size - 1) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
    k /= k.sum() + 1e-9
    return cv2.filter2D(img, -1, k)


def _gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise. Input/output: uint8 RGB."""
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _extreme_day_night(img: np.ndarray) -> np.ndarray:
    """Simulates severe nighttime underexposure (crushed blacks) or daytime blown-out highlights."""
    # 50% chance for dark/night, 50% chance for harsh overexposure
    gamma = random.uniform(2.5, 4.0) if random.random() < 0.5 else random.uniform(0.2, 0.6)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def _generate_planar_homography_pair(img_t_rgb: np.ndarray, img_s_rgb: np.ndarray,
                                     raw_dx: float, raw_dy: float,
                                     max_skew: float = 0.35, max_disp: float = 28.0,
                                     _retries: int = 0):
    """
    Simulates HPatches Viewpoint changes by keeping the original 128px template untouched,
    but applying a severe homography warp to the 192px search image.
    Tracks the true Ground Truth pixel perfectly through the non-linear warp.
    """
    h, w = img_s_rgb.shape[:2]

    # 1. Create the aggressive homography skew matrix
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = pts1 + np.random.uniform(-max_skew * w, max_skew * w, pts1.shape).astype(np.float32)
    H = cv2.getPerspectiveTransform(pts1, pts2)

    # 2. Track the ORIGINAL GT point through the warp H
    gt_orig_x = 96.0 + raw_dx
    gt_orig_y = 96.0 + raw_dy
    pt = np.array([[[gt_orig_x, gt_orig_y]]], dtype=np.float32)

    pt_warped = cv2.perspectiveTransform(pt, H)[0][0]
    new_dx = pt_warped[0] - 96.0
    new_dy = pt_warped[1] - 96.0

    # 3. Safe Boundary Guard: DO NOT CLIP. Retry if the point is pushed out of the 128px crop limit.
    if abs(new_dx) > max_disp or abs(new_dy) > max_disp:
        if _retries < 5:
            # Reduce skew by 20% and try again
            return _generate_planar_homography_pair(
                img_t_rgb, img_s_rgb, raw_dx, raw_dy,
                max_skew=max_skew * 0.8, max_disp=max_disp, _retries=_retries + 1)
        else:
            # Fallback: Return original images and labels if we can't find a safe warp
            return img_t_rgb.copy(), img_s_rgb.copy(), raw_dx, raw_dy

    # 4. Warp ONLY the search image (leaving the template as the perfect "reference" view)
    warped_s = cv2.warpPerspective(img_s_rgb, H, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return img_t_rgb.copy(), warped_s, float(new_dx), float(new_dy)

class RefinementDataset(Dataset):
    def __init__(self, root_dir, split="train", scale_factor=16.0, max_jitter=5.0,
                 crop_search_to_128: bool = True):
        """
        Args:
            root_dir (str): Directory with all the images and labels.csv.
            split (str): "train" or "val". Determines if augmentations are applied.
            scale_factor (float): Normalization factor for dx/dy.
            max_jitter (float): Max pixel shift applied during training to simulate tracker noise.
                                Only used when crop_search_to_128=True.
            crop_search_to_128 (bool): If True (default, used for training), crops the on-disk
                                       192px search image to 128px with optional jitter.
                                       If False (used for multi-iter validation), returns the
                                       full on-disk search image so the forward pass can
                                       iterate over fresh 128px crops internally.
        """
        self.split_dir = os.path.join(root_dir, split)
        self.labels_path = os.path.join(self.split_dir, "labels.csv")
        self.scale_factor = scale_factor
        self.split = split
        self.max_jitter = max_jitter
        self.crop_search_to_128 = crop_search_to_128

        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"CSV file not found: {self.labels_path}")

        self.data = pd.read_csv(self.labels_path)

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Shared color augmentation (applied with same seed to both images)
        self.augment_shared = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]

        # Load Images
        img_t = cv2.imread(os.path.join(self.split_dir, row['filename_template']))
        img_s = cv2.imread(os.path.join(self.split_dir, row['filename_search']))

        if img_t is None or img_s is None:
            raise FileNotFoundError(f"Failed to load image at index {idx}")

        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

        raw_dx, raw_dy = float(row['dx']), float(row['dy'])
        is_occluded = float(row['occluded']) if 'occluded' in self.data.columns else 0.0

        # ===================================================================
        # THE HOMOGRAPHY HACK (25% of the time during training)
        # We discard the 3D template and generate a mathematically perfect 2D skew.
        # ===================================================================
        if self.split == "train" and random.random() < 0.25:
            # img_t (real 3D viewpoint) gets a mild warp; img_s gets a strong warp.
            # GT is recomputed by tracking the original GT point through the search warp.
            img_t, img_s, raw_dx, raw_dy = _generate_planar_homography_pair(
                img_t, img_s, raw_dx, raw_dy, max_skew=0.35)
            # Planar projection — mathematically never occluded
            is_occluded = 0.0
        # ===================================================================

        if self.split == "train":
            # --- Extreme Day/Night Augmentation ---
            # Apply to ONLY ONE of the images to force the network to bridge the gap
            if random.random() < 0.2:
                if random.random() < 0.5:
                    img_t = _extreme_day_night(img_t)
                else:
                    img_s = _extreme_day_night(img_s)

            # JPEG compression
            if random.random() < 0.5:
                img_t = _jpeg_compress(img_t, random.randint(30, 70))
            if random.random() < 0.5:
                img_s = _jpeg_compress(img_s, random.randint(30, 70))

            # Motion blur
            if random.random() < 0.3:
                img_s = _motion_blur(img_s, random.choice([3, 5, 7]), random.uniform(0, 360))

            # Gaussian noise
            if random.random() < 0.4:
                img_t = _gaussian_noise(img_t, random.uniform(3, 15))
            if random.random() < 0.4:
                img_s = _gaussian_noise(img_s, random.uniform(3, 15))

        pil_t, pil_s = Image.fromarray(img_t), Image.fromarray(img_s)

        # --- Shared color augmentations (same random params for both) ---
        if self.split == "train":
            seed = torch.randint(0, 2 ** 31, (1,)).item()
            random.seed(seed)
            torch.manual_seed(seed)
            pil_t = self.augment_shared(pil_t)
            random.seed(seed)
            torch.manual_seed(seed)
            pil_s = self.augment_shared(pil_s)

            # --- Independent brightness/contrast shift (breaks shared-seed symmetry) ---
            # Simulates different exposure/lighting between frames.
            if random.random() < 0.4:
                pil_t = TF.adjust_brightness(pil_t, random.uniform(0.8, 1.2))
            if random.random() < 0.4:
                pil_s = TF.adjust_brightness(pil_s, random.uniform(0.8, 1.2))
            if random.random() < 0.4:
                pil_t = TF.adjust_contrast(pil_t, random.uniform(0.8, 1.2))
            if random.random() < 0.4:
                pil_s = TF.adjust_contrast(pil_s, random.uniform(0.8, 1.2))

        # --- Scale jitter (train only): simulate zoom/depth change between frames ---
        # Randomly rescale the search patch to teach the network to handle cases where
        # the tracked object appears at a different scale in the search vs template
        # (e.g., camera/subject moving in depth between frames in real video).
        # GT offsets scale linearly with the zoom factor.
        scale_s = 1.0
        if self.split == "train":
            # Light jitter only (±7%): the main zoom diversity comes from the v6 dataset
            # (FOV-based zoom baked into 50% of saved pairs). This small online jitter adds
            # epoch-to-epoch stochasticity for the non-zoomed 50% without stacking aggressively
            # on top of already-zoomed pairs (worst case: 1.33 × 1.07 ≈ 1.42×).
            scale_s = random.uniform(0.93, 1.07)
            if abs(scale_s - 1.0) > 0.01:
                orig_w, orig_h = pil_s.size  # PIL: (W, H)
                new_w = max(1, int(round(orig_w * scale_s)))
                new_h = max(1, int(round(orig_h * scale_s)))
                pil_s = TF.resize(pil_s, [new_h, new_w])
                # center_crop pads with zeros when new size < original (zoom-out case)
                pil_s = TF.center_crop(pil_s, [orig_h, orig_w])

        # Convert to Normalized Tensors
        tensor_t, tensor_s = self.normalize(pil_t), self.normalize(pil_s)

        # Apply scale: target moves proportionally within the rescaled search patch
        # (raw_dx/raw_dy are already set — either from CSV or from the homography hack above)
        raw_dx *= scale_s
        raw_dy *= scale_s

        # --- CROP TO 128px (training) or return full image (multi-iter val) ---
        _, h_s, w_s = tensor_s.shape
        if self.crop_search_to_128 and (h_s > 128 or w_s > 128):
            # Extract a centered 128px crop. Jitter shifts the crop during training so
            # the model learns to correct off-center initializations. GT adjusted accordingly.
            jx = random.uniform(-self.max_jitter, self.max_jitter) if (self.split == "train" and self.max_jitter > 0) else 0.0
            jy = random.uniform(-self.max_jitter, self.max_jitter) if (self.split == "train" and self.max_jitter > 0) else 0.0

            top  = int(round((h_s - 128) / 2 + jy))
            left = int(round((w_s - 128) / 2 + jx))
            tensor_s = TF.crop(tensor_s, top, left, 128, 128)

            # Shifting the crop window by +jx means the target moves -jx relative to the window
            raw_dx -= jx
            raw_dy -= jy
        elif not self.crop_search_to_128 and h_s == 128 and w_s == 128:
            # Real-data images are 128px but multi-iter validation needs 192px so the
            # forward() iterative branch triggers (condition: h_s > 128 or w_s > 128).
            # Center-embed the 128px crop in a 192px canvas (32px border on each side).
            # The label (dx, dy) stays unchanged: the target displacement from the 128px
            # center equals the displacement from the new 192px center (both at pixel 96).
            tensor_s = TF.pad(tensor_s, padding=32)  # 128 + 32*2 = 192px
        # Otherwise crop_search_to_128=False and image is already >128px (192px synthetic):
        # return as-is; forward() handles iterative cropping internally.

        # Normalize Labels
        label = torch.tensor([raw_dx, raw_dy], dtype=torch.float32) / self.scale_factor

        # Occlusion handling (is_occluded already set above — 0.0 for homography samples)
        gt_visible = torch.tensor(1.0 - is_occluded, dtype=torch.float32)

        return tensor_t, tensor_s, label, gt_visible


# ---------------------------------------------------------------------------
# Quick visual debugger — run directly to inspect augmentations
#   python RefinementDataset.py [--root PATH] [--n 30] [--seed 0]
#   SPACE = next sample   Q = quit
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _to_bgr(tensor: "torch.Tensor") -> np.ndarray:
        """Denormalize tensor [C,H,W] → uint8 BGR for cv2."""
        img = tensor.permute(1, 2, 0).numpy()       # [H, W, C] float32
        img = (img * _STD + _MEAN).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _draw_cross(img: np.ndarray, cx: float, cy: float,
                    color: tuple, size: int = 10, thickness: int = 2) -> None:
        x, y = int(round(cx)), int(round(cy))
        cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

    p = argparse.ArgumentParser(description="Visualize RefinementDataset augmentations")
    p.add_argument("--root", type=str,
                   default="./datasets/synth_train",
                   help="Dataset root dir (contains train/ and val/ sub-dirs with labels.csv)")
    p.add_argument("--n",    type=int, default=50,  help="Number of samples to browse")
    p.add_argument("--seed", type=int, default=0,   help="Shuffle seed")
    cfg = p.parse_args()

    # Try to standardize path if my_utils is available
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from subpixr.utils import standardize_path
        cfg.root = standardize_path(cfg.root)
    except ImportError:
        pass

    dataset = RefinementDataset(cfg.root, split="val", scale_factor=8.0,
                                max_jitter=15.0, crop_search_to_128=True)
    rng = random.Random(cfg.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:cfg.n]

    print(f"Dataset: {len(dataset)} samples  |  browsing {len(indices)}")
    print("Controls: SPACE = next   Q = quit")

    WIN = "RefinementDataset — SPACE next  Q quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    for sample_i, idx in enumerate(indices):
        tensor_t, tensor_s, label, gt_visible = dataset[idx]

        t_bgr = _to_bgr(tensor_t)                         # 128×128
        s_bgr = _to_bgr(tensor_s)                         # 128×128 (or 192 if crop=False)

        # Scale label back to pixels to find GT position in the search crop
        scale_factor = 8.0
        gt_dx = float(label[0]) * scale_factor
        gt_dy = float(label[1]) * scale_factor
        sh, sw = s_bgr.shape[:2]
        cx_s = sw / 2.0 + gt_dx
        cy_s = sh / 2.0 + gt_dy

        # Draw GT cross (green) and search-center mark (blue) on the search patch
        s_vis = s_bgr.copy()
        _draw_cross(s_vis, sw / 2.0, sh / 2.0, (255, 80,  0),  size=8)   # blue = center/coarse
        _draw_cross(s_vis, cx_s,     cy_s,      (0,   220, 0),  size=6)   # green = GT

        # Draw center mark on template (the thing we're trying to match)
        t_vis = t_bgr.copy()
        _draw_cross(t_vis, 64, 64, (0, 220, 0), size=6)

        # Pad search to 128px if it's larger, for side-by-side display
        if sh != 128 or sw != 128:
            s_vis = cv2.resize(s_vis, (128, 128))

        # Determine augmentation type from label magnitude (homography → large GT offset)
        aug_note = ""
        disp = (gt_dx ** 2 + gt_dy ** 2) ** 0.5
        if disp < 0.1:
            aug_note = "HOMOGRAPHY-FALLBACK(dx=0)"
        elif disp > 20:
            aug_note = f"LARGE DISP({disp:.1f}px)"

        # Side-by-side panel: [template | search] + text info
        panel = np.hstack([t_vis, s_vis])

        row = dataset.data.iloc[idx]
        occ_str = "occluded" if float(label[0]) == 0 and float(gt_visible) < 0.5 else ""
        info = (f"#{sample_i+1}/{len(indices)}  idx={idx}  "
                f"GT=({gt_dx:+.1f},{gt_dy:+.1f})px  "
                f"vis={float(gt_visible):.0f}  {aug_note}  {occ_str}")
        # Add text bar below panel
        bar = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, info, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        # Second line: filenames
        fnames = f"T:{row.get('filename_template','?')}   S:{row.get('filename_search','?')}"
        bar2 = np.zeros((22, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar2, fnames, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)

        display = np.vstack([panel, bar, bar2])
        cv2.imshow(WIN, display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Done.")