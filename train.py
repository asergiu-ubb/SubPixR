import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

wandb_enabled = not sys.platform.startswith('win') and os.environ.get("WANDB_API_KEY")
if wandb_enabled:
    import wandb
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from subpixr.utils import change_path_for_linux
from data.refinement_dataset import RefinementDataset
from subpixr.model import RefinementNetwork


def sequence_loss(criterion, predictions, label_gt, valid_mask, gamma=0.8):
    """RAFT-style per-iteration loss with exponentially increasing weights.
    predictions: list of (B, 2) accumulated deltas, one per iteration.
    Weight for iteration i: gamma^(N-1-i), so last iteration has weight 1.0.
    """
    n = len(predictions)
    total_loss = torch.tensor(0.0, device=label_gt.device)
    for i, pred in enumerate(predictions):
        weight = gamma ** (n - 1 - i)
        if valid_mask.any():
            iter_loss = criterion(pred[valid_mask], label_gt[valid_mask])
        else:
            iter_loss = (pred * 0).sum()
        total_loss = total_loss + weight * iter_loss
    return total_loss


CUDA_CHECK_INTERVAL = 50  # check GPU health every N batches inside the training loop


def train_epoch(model, loader, optimizer, criterion_coord, criterion_conf, train_iters=1):
    model.train()
    running_loss, running_acc = 0.0, 0.0

    loop = tqdm(loader, leave=False)
    for batch_idx, (template, search, label_gt, gt_visible) in enumerate(loop):
        if batch_idx % CUDA_CHECK_INTERVAL == 0 and batch_idx > 0:
            check_cuda_health(f"batch {batch_idx}")

        template, search, label_gt, gt_visible = [x.to(CONFIG["device"]) for x in
                                                  (template, search, label_gt, gt_visible)]
        optimizer.zero_grad()

        if train_iters > 1:
            if CONFIG["predict_confidence"] and not CONFIG["use_pmr_confidence"]:
                # Grab the sequence of deltas AND the raw output of the final iteration
                predictions, raw_out = model(template, search, num_iters=train_iters, return_intermediate=True)

                valid_mask = gt_visible.bool() & (torch.norm(label_gt, dim=-1) < CONFIG["max_valid_dist"])
                loss_coord = sequence_loss(criterion_coord, predictions, label_gt, valid_mask, gamma=0.8)

                # Calculate Confidence Target based on the FINAL accumulated delta
                pred_conf_logit = raw_out[:, 2]
                final_pred_coords = predictions[-1]

                with torch.no_grad():
                    # Error is in normalized coordinates. Multiply by scale_factor to get pixels
                    final_err_px = torch.norm(label_gt - final_pred_coords.detach(), dim=-1) * CONFIG["scale_factor"]
                    # If the prediction is within 1 pixel of the GT, we should be confident (1.0)
                    gt_conf = (final_err_px < 1.0).float()

                loss_conf = criterion_conf(pred_conf_logit, gt_conf)
                loss = loss_coord + (CONFIG["conf_weight"] * loss_conf)

                pred_class = (pred_conf_logit > 0.0).float()
                acc = (pred_class == gt_conf).float().mean().item()
                running_acc += acc
            else:
                predictions = model(template, search, num_iters=train_iters, return_intermediate=True)
                valid_mask = gt_visible.bool() & (torch.norm(label_gt, dim=-1) < CONFIG["max_valid_dist"])
                loss = sequence_loss(criterion_coord, predictions, label_gt, valid_mask, gamma=0.8)
                acc = 0.0
        else:
            pred = model(template, search, num_iters=train_iters)
            valid_mask = gt_visible.bool() & (torch.norm(label_gt, dim=-1) < CONFIG["max_valid_dist"])

            if CONFIG["predict_confidence"] and not CONFIG["use_pmr_confidence"]:
                pred_coords = pred[:, :2]
                pred_conf_logit = pred[:, 2]

                loss_coord = criterion_coord(pred_coords[valid_mask], label_gt[valid_mask]) if valid_mask.any() else (
                            pred_coords * 0).sum()

                with torch.no_grad():
                    final_err_px = torch.norm(label_gt - pred_coords.detach(), dim=-1) * CONFIG["scale_factor"]
                    gt_conf = (final_err_px < 1.0).float()

                loss_conf = criterion_conf(pred_conf_logit, gt_conf)
                loss = loss_coord + (CONFIG["conf_weight"] * loss_conf)

                pred_class = (pred_conf_logit > 0.0).float()
                acc = (pred_class == gt_conf).float().mean().item()
                running_acc += acc
            else:
                loss = criterion_coord(pred[:, :2][valid_mask], label_gt[valid_mask]) if valid_mask.any() else (
                            pred * 0).sum()
                acc = 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(
            f"Loss: {loss.item():.4f}" + (f" | Conf Acc: {acc * 100:.1f}%" if CONFIG["predict_confidence"] else ""))

    return running_loss / len(loader), running_acc / len(loader)


def val_epoch(model, loader, criterion_coord, criterion_conf, scale_factor, num_iters=1):
    model.eval()
    running_loss, running_acc, total_pixel_error, valid_samples_count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for template, search, label_gt, gt_visible in loader:
            template, search, label_gt, gt_visible = [x.to(CONFIG["device"]) for x in
                                                      (template, search, label_gt, gt_visible)]

            # Because model.eval() is set, the network automatically applies the sigmoid gate internally
            # We need to temporarily force it to return raw logits if we want to calculate val loss

            if CONFIG["predict_confidence"] and not CONFIG["use_pmr_confidence"] and num_iters == 1:
                # Use the new return_logits flag instead of hacking model.train()
                raw_pred = model(template, search, num_iters=1, return_logits=True)

                pred_coords = raw_pred[:, :2]
                pred_conf_logit = raw_pred[:, 2]

                # Recreate the gated output for pixel error evaluation
                conf = torch.sigmoid(pred_conf_logit).unsqueeze(1)
                pred_delta_px = (pred_coords * conf) * scale_factor

                valid_mask = gt_visible.bool() & (torch.norm(label_gt, dim=-1) < (128.0 / scale_factor))
                loss_coord = criterion_coord(pred_coords[valid_mask], label_gt[valid_mask]) if valid_mask.any() else 0.0

                # Calc confidence accuracy using the new Absolute Threshold (< 1.0 px)
                final_err_px = torch.norm(label_gt - pred_coords, dim=-1) * scale_factor
                gt_conf = (final_err_px < 1.0).float()

                loss_conf = criterion_conf(pred_conf_logit, gt_conf)

                loss = loss_coord + (0.1 * loss_conf)
                running_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

                pred_class = (pred_conf_logit > 0.0).float()
                running_acc += (pred_class == gt_conf).float().mean().item()

            elif CONFIG.get("use_pmr_confidence", False) and num_iters == 1:
                # Val loss on raw (ungated) delta; pixel error uses confidence-gated output.
                raw_pred = model(template, search, num_iters=1, return_logits=True)  # [B, 3]
                pred_coords = raw_pred[:, :2]
                pred_conf = raw_pred[:, 2:3]
                pred_delta_px = (pred_coords * pred_conf) * scale_factor  # gated, for pixel error

                valid_mask = gt_visible.bool() & (torch.norm(label_gt, dim=-1) < (128.0 / scale_factor))
                loss = criterion_coord(pred_coords[valid_mask], label_gt[valid_mask]) if valid_mask.any() else (pred_coords * 0).sum()
                running_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

            else:
                pred = model(template, search, num_iters=num_iters)
                pred_delta_px = pred[:, :2] * scale_factor

                valid_mask = torch.norm(label_gt, dim=-1) < 128.0
                loss = criterion_coord(pred[:, :2], label_gt)
                running_loss += loss.item()

            if valid_mask.sum() > 0:
                label_gt_px = label_gt[valid_mask] * scale_factor
                error = torch.sqrt(torch.sum((pred_delta_px[valid_mask] - label_gt_px) ** 2, dim=1))
                total_pixel_error += error.sum().item()
                valid_samples_count += valid_mask.sum().item()

    return running_loss / len(loader), (total_pixel_error / valid_samples_count if valid_samples_count > 0 else 0.0), (
        running_acc / len(loader) if CONFIG["predict_confidence"] else 0.0)


def check_cuda_health(context: str = ""):
    """Probe the GPU at driver + runtime level."""
    if CONFIG["device"] != "cuda":
        return
    tag = f" [{context}]" if context else ""
    try:
        if torch.cuda.device_count() == 0:
            raise RuntimeError("device_count() == 0 — NVML reports no GPUs")
        torch.cuda.get_device_properties(0)
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as e:
        print(f"\n[FATAL]{tag} CUDA health check failed: {e}")
        print("[FATAL] GPU lost (NVML/driver error). Stopping — restart the Docker container.")
        sys.exit(1)


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    train_iters = CONFIG["train_iters"]

    # 2. Helper to ensure T/F strings and consistent float formatting
    def b2s(val):
        return "True" if val else "False"

    val_iters = CONFIG["val_iters"]
    # 3. The New Canonical Run Name structure
    run_name = (
        f"v7.1_{timestamp}_{CONFIG['encoder_type']}_"
        f"xcorr_{b2s(CONFIG['use_depthwise_xcorr'])}_"
        f"attn_{b2s(CONFIG['use_attention'])}_"
        f"conf_{b2s(CONFIG['predict_confidence'])}_"
        f"cost_{b2s(CONFIG['use_local_cost_volume'])}_"
        f"spatial_{b2s(CONFIG['use_spatial_head'])}_"
        f"multi_{b2s(CONFIG['use_multi_stage_features'])}_"
        f"hybrid_{b2s(CONFIG['use_hybrid_fusion'])}_"
        f"jit_{CONFIG['max_jitter']:.1f}_"
        f"scale{CONFIG['scale_factor']:.1f}_"
        f"ti{train_iters}_"
        f"vi{val_iters}"
    )
    run_save_dir = os.path.join(CONFIG["base_save_dir"], run_name)
    os.makedirs(run_save_dir, exist_ok=True)

    # Write absolute checkpoint path so shell scripts can find it after training
    _abs_save_dir = os.path.abspath(run_save_dir)
    with open(os.path.join(CONFIG["base_save_dir"], "last_run_dir.txt"), "w") as _f:
        _f.write(_abs_save_dir)
    print(f"[train_refiner] Checkpoint dir: {_abs_save_dir}")

    with open(os.path.join(run_save_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
    if wandb_enabled: wandb.init(project=CONFIG["project_name"], name=run_name, config=CONFIG)

    check_cuda_health()
    print("Loading datasets...")
    crop_train = (train_iters == 1)
    train_ds = RefinementDataset(DATA_ROOT, "train", CONFIG["scale_factor"], CONFIG["max_jitter"],
                                 crop_search_to_128=crop_train)
    val_ds = RefinementDataset(DATA_ROOT, "val", CONFIG["scale_factor"], 0.0)

    val_domain_ds = RefinementDataset(VAL_DOMAIN, "", CONFIG["scale_factor"], 0.0)
    val_domain_ds_full = RefinementDataset(VAL_DOMAIN, "", CONFIG["scale_factor"], 0.0, crop_search_to_128=False)
    val_davis_ds = RefinementDataset(VAL_DAVIS, "", CONFIG["scale_factor"], 0.0)
    val_rgb_stack_ds = RefinementDataset(VAL_RGB_STACKING, "", CONFIG["scale_factor"], 0.0)
    val_megadepth_ds = RefinementDataset(VAL_MEGADEPTH, "", CONFIG["scale_factor"], 0.0)
    val_hpatches_ds = RefinementDataset(VAL_HPATCHES, "", CONFIG["scale_factor"], 0.0)

    train_bs = CONFIG["batch_size"] // train_iters if train_iters > 1 else CONFIG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    val_domain_loader = DataLoader(val_domain_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                                 num_workers=CONFIG["num_workers"])
    val_domain_loader_full = DataLoader(val_domain_ds_full, batch_size=CONFIG["batch_size"], shuffle=False,
                                      num_workers=CONFIG["num_workers"])
    val_davis_loader = DataLoader(val_davis_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                                  num_workers=CONFIG["num_workers"])
    val_rgb_stack_loader = DataLoader(val_rgb_stack_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                                      num_workers=CONFIG["num_workers"])
    val_megadepth_loader = DataLoader(val_megadepth_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                                      num_workers=CONFIG["num_workers"])
    val_hpatches_loader = DataLoader(val_hpatches_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                                     num_workers=CONFIG["num_workers"])

    model = RefinementNetwork(
        encoder_type=CONFIG["encoder_type"], freeze_encoder=CONFIG["freeze_encoder"],
        dropout_rate=CONFIG["dropout_rate"],
        use_depthwise_xcorr=CONFIG["use_depthwise_xcorr"],
        use_attention=CONFIG["use_attention"],
        use_spatial_head=CONFIG["use_spatial_head"],
        predict_confidence=CONFIG["predict_confidence"],
        use_local_cost_volume=CONFIG["use_local_cost_volume"],
        scale_factor=CONFIG["scale_factor"],
        use_multi_stage_features=CONFIG["use_multi_stage_features"],
        use_pmr_confidence=CONFIG["use_pmr_confidence"],
        use_hybrid_fusion=CONFIG["use_hybrid_fusion"]  # <-- ADDED
    ).to(CONFIG["device"])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG["epochs"] // 2, gamma=0.5)

    criterion_coord = nn.SmoothL1Loss(beta=0.1).to(CONFIG["device"])
    criterion_conf = nn.BCEWithLogitsLoss().to(CONFIG["device"])

    best_composite_error, best_epoch = float("inf"), 0
    best_val_davis_error = float("inf")
    best_val_megadepth_error = float("inf")
    best_val_domain_error = float("inf")
    best_val_rgb_stack_error = float("inf")
    best_val_hpatches_error = float("inf")

    print(f"Starting training... (train_iters={train_iters}, train_bs={train_bs})")
    for epoch in range(CONFIG["epochs"]):
        check_cuda_health()
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion_coord, criterion_conf,
                                            train_iters=train_iters)
        val_loss, val_pixel_error, val_acc = val_epoch(model, val_loader, criterion_coord, criterion_conf,
                                                       CONFIG["scale_factor"])
        _, val_domain_pixel_error, _ = val_epoch(model, val_domain_loader, criterion_coord, criterion_conf,
                                               CONFIG["scale_factor"], num_iters=1)
        _, val_davis_pixel_error, _ = val_epoch(model, val_davis_loader, criterion_coord, criterion_conf,
                                                CONFIG["scale_factor"], num_iters=1)
        _, val_rgb_stack_pixel_error, _ = val_epoch(model, val_rgb_stack_loader, criterion_coord, criterion_conf,
                                                    CONFIG["scale_factor"], num_iters=1)
        _, val_megadepth_pixel_error, _ = val_epoch(model, val_megadepth_loader, criterion_coord, criterion_conf,
                                                    CONFIG["scale_factor"], num_iters=1)
        _, val_hpatches_pixel_error, _ = val_epoch(model, val_hpatches_loader, criterion_coord, criterion_conf,
                                                   CONFIG["scale_factor"], num_iters=1)

        scheduler.step()

        # Added HPatches to the composite error and adjusted divisor to 5.0
        val_composite_error = (
                                          val_davis_pixel_error + 2 * val_megadepth_pixel_error + val_domain_pixel_error + val_hpatches_pixel_error) / 5.0

        best_val_davis_error = min(best_val_davis_error, val_davis_pixel_error)
        best_val_megadepth_error = min(best_val_megadepth_error, val_megadepth_pixel_error)
        best_val_domain_error = min(best_val_domain_error, val_domain_pixel_error)
        best_val_rgb_stack_error = min(best_val_rgb_stack_error, val_rgb_stack_pixel_error)
        best_val_hpatches_error = min(best_val_hpatches_error, val_hpatches_pixel_error)

        if val_composite_error < best_composite_error:
            best_composite_error, best_epoch = val_composite_error, epoch + 1
            torch.save(model.state_dict(), os.path.join(run_save_dir, "best_model.pth"))
            print("--> New best model saved!")

        if wandb_enabled:
            wandb.log({
                "train/loss": train_loss,
                "train/conf_acc": train_acc,
                "val/loss": val_loss,
                "val/conf_acc": val_acc,
                "val/domain_pixel_error": val_domain_pixel_error,
                "val/davis_pixel_error": val_davis_pixel_error,
                "val/rgb_stacking_pixel_error": val_rgb_stack_pixel_error,
                "val/megadepth_pixel_error": val_megadepth_pixel_error,
                "val/hpatches_pixel_error": val_hpatches_pixel_error,
                "val/composite_error": val_composite_error
            })

        print(f"Train Loss: {train_loss:.4f} | Val Err: {val_pixel_error:.4f} px "
              f"| Domain: {val_domain_pixel_error:.4f} px "
              f"| DAVIS: {val_davis_pixel_error:.4f} px "
              f"| RGB-Stack: {val_rgb_stack_pixel_error:.4f} px "
              f"| MegaDepth: {val_megadepth_pixel_error:.4f} px "
              f"| HPatches: {val_hpatches_pixel_error:.4f} px "
              f"| Composite: {val_composite_error:.4f} px")

    torch.save(model.state_dict(), os.path.join(run_save_dir, "last_model.pth"))

    print(f"\nRunning {CONFIG['val_iters']}-iter validation on best model...")
    model.load_state_dict(torch.load(os.path.join(run_save_dir, "best_model.pth")))
    _, val_domain_pixel_error_Niter, _ = val_epoch(model, val_domain_loader_full, criterion_coord, criterion_conf,
                                                 CONFIG["scale_factor"], num_iters=CONFIG["val_iters"])
    print(
        f"Best Composite: {best_composite_error:.4f} px (epoch {best_epoch}) | Best Domain {CONFIG['val_iters']}-iter: {val_domain_pixel_error_Niter:.4f} px")

    model.eval()
    _device = CONFIG["device"]
    dummy_t = torch.randn(1, 3, 128, 128, device=_device)
    dummy_s = torch.randn(1, 3, 128, 128, device=_device)
    with torch.no_grad():
        for _ in range(20):
            model(dummy_t, dummy_s, num_iters=1)
    N_BENCH = 200
    if _device == "cuda":
        torch.cuda.synchronize()
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        with torch.no_grad():
            for _ in range(N_BENCH):
                model(dummy_t, dummy_s, num_iters=1)
        t_end.record()
        torch.cuda.synchronize()
        inference_time_ms = t_start.elapsed_time(t_end) / N_BENCH
    else:
        import time as _time
        _t0 = _time.perf_counter()
        with torch.no_grad():
            for _ in range(N_BENCH):
                model(dummy_t, dummy_s, num_iters=1)
        inference_time_ms = (_time.perf_counter() - _t0) / N_BENCH * 1000.0
    print(f"Inference time (bs=1, 1-iter): {inference_time_ms:.3f} ms")

    if wandb_enabled:
        wandb.summary["best_val_davis_pixel_error"] = best_val_davis_error
        wandb.summary["best_val_epoch"] = best_epoch
        wandb.summary[f"val/domain_pixel_error_{CONFIG['val_iters']}iter"] = val_domain_pixel_error_Niter
        wandb.summary["val/davis_pixel_error_last_epoch"] = val_davis_pixel_error
        wandb.summary["val/rgb_stacking_pixel_error_last_epoch"] = val_rgb_stack_pixel_error
        wandb.summary["val/megadepth_pixel_error_last_epoch"] = val_megadepth_pixel_error
        wandb.summary["val/hpatches_pixel_error_last_epoch"] = val_hpatches_pixel_error
        wandb.summary["inference_time_ms"] = inference_time_ms
        wandb.finish()
    print(f"Training Complete. Best Composite: {best_composite_error:.4f} px at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', type=str, default='resnet18')

    parser.add_argument('--use_depthwise_xcorr', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_attention', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--predict_confidence', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_pmr_confidence', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_local_cost_volume', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_spatial_head', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_multi_stage_features', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_hybrid_fusion', type=lambda x: (str(x).lower() == 'true'), default=False) # <-- ADDED
    parser.add_argument('--scale_factor', type=float, default=16.0)

    parser.add_argument('--max_jitter', type=float, default=15.0)
    parser.add_argument('--train_iters', type=int, default=4)
    parser.add_argument('--val_iters', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=2 if sys.platform.startswith('win') else 32)
    parser.add_argument('--coord_weight', type=float, default=1.0)
    parser.add_argument('--conf_weight', type=float, default=0.1)
    parser.add_argument('--max_valid_dist', type=float, default=8.0)

    args = parser.parse_args()

    CONFIG = {
        "project_name": os.environ.get("WANDB_PROJECT", "subpixr"),
        "encoder_type": args.encoder_type,
        "use_depthwise_xcorr": args.use_depthwise_xcorr,
        "use_attention": args.use_attention,
        "predict_confidence": args.predict_confidence,
        "use_pmr_confidence": args.use_pmr_confidence,
        "use_local_cost_volume": args.use_local_cost_volume,
        "use_spatial_head": args.use_spatial_head,
        "use_multi_stage_features": args.use_multi_stage_features,
        "use_hybrid_fusion": args.use_hybrid_fusion,
        "scale_factor": args.scale_factor,
        "max_jitter": args.max_jitter,
        "train_iters": args.train_iters,
        "val_iters": args.val_iters,
        "batch_size": args.batch_size,
        "coord_weight": args.coord_weight,
        "conf_weight": args.conf_weight,
        "max_valid_dist": args.max_valid_dist,
        "learning_rate": 1e-4,
        "epochs": 50,
        "num_workers": 0 if sys.platform.startswith('win') else 24,
        "freeze_encoder": False,
        "dropout_rate": 0.4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "base_save_dir": "./checkpoints"
    }

    # Dataset roots — override via DATASETS env var, otherwise read from ./datasets.
    # The synthetic training data is produced by data/generate_synthetic_dataset.py.
    # Validation roots are RefinementDataset-format exports of public benchmarks.
    # VAL_DOMAIN points at an optional domain-specific real-data validation set;
    # if unset, falls back to DAVIS so the training script remains runnable.
    _DS_ROOT = os.environ.get("DATASETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"))
    DATA_ROOT        = os.path.join(_DS_ROOT, "synth_train")
    VAL_DAVIS        = os.path.join(_DS_ROOT, "val_davis")
    VAL_RGB_STACKING = os.path.join(_DS_ROOT, "val_rgb_stacking")
    VAL_MEGADEPTH    = os.path.join(_DS_ROOT, "val_megadepth")
    VAL_HPATCHES     = os.path.join(_DS_ROOT, "val_hpatches")
    VAL_DOMAIN       = os.environ.get("VAL_DOMAIN", VAL_DAVIS)

    main()