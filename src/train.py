"""
Training code for the cosmic void 3D detector.

Refactored into composable functions:
  process_head  → single head, single sample
  process_sample → all heads for one sample
  process_batch  → forward + loss for one batch
  run_epoch      → full train or val epoch
  train_detector → outer training loop

Plus:
  TrainingConfig   → all hyperparameters in one place
  ModelCheckpointer → save/load logic
"""
import os
import signal
import time
from dataclasses import dataclass

import pandas as pd
import torch
import torch.onnx
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from src.profiler import ResourceProfiler

from src.losses import YOLO3DLoss
from src.model import CosmicVoidDetectionVNet, Config3D
from src.dataset import void_data_loader, target_to_voids
from src.metrics import compute_cen_rad_iou_prec_rec_f1, compute_accuracy
from src.train_logger import TrainingLogger
from src.full_grid_tensor import group_means, BoxClusterer
from src.filter_overlap import filter_touching_spheres
from src.dcn_config import get_output_paths, SCALE_ADJ

OUT_FOLDER = get_output_paths()


# ──────────────────────────────────────────────
# Step 6: TrainingConfig
# ──────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Holds all training hyper-parameters in one place."""
    lr: float = 0.00005
    epochs: int = 10
    batch_size: int = 1
    weight_decay: float = 0.01
    hwem: int = 2  # head weight exponential multiplier
    n_anchors: int = 3
    cube_size: float = 2000 / 16


# ──────────────────────────────────────────────
# Step 1: ModelCheckpointer
# ──────────────────────────────────────────────
class ModelCheckpointer:
    """Handles saving, loading, and best-model tracking."""

    def __init__(self, out_folder, exp_name):
        """Initialize the checkpointer with output folder and experiment name."""
        self.out_folder = out_folder
        self.exp_name = exp_name
        self.best_loss = torch.inf

    def _path(self, filename):
        """Construct the full path for a given checkpoint filename."""
        return os.path.join(self.out_folder, self.exp_name, filename)

    def save_if_best(self, model, loss):
        """Save the model if the given loss is better than the best seen so far."""
        loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
        if loss_val < self.best_loss:
            torch.save(model.state_dict(), self._path("Best_model.pth"))
            self.best_loss = loss_val

    def save_last(self, model):
        """Save the last model state (overwrites every epoch)."""
        torch.save(model.state_dict(), self._path("last_model.pth"))

    @staticmethod
    def find_latest_experiment(out_folder):
        """Return the most recent experiment folder name, or None."""
        try:
            folders = sorted(
                f
                for f in os.listdir(out_folder)
                if os.path.isdir(os.path.join(out_folder, f))
                and os.path.exists(
                    os.path.join(out_folder, f, "last_model (copia).pth")
                )
            )
            return folders[-1] if folders else None
        except FileNotFoundError:
            return None

    @staticmethod
    def load_or_init(model, out_folder, last_exp="auto"):
        """
        Try full load → partial load (strict=False) → fresh init.
        If last_exp="auto", picks the latest folder containing Best_model.pth.
        """
        if last_exp == "auto":
            last_exp = ModelCheckpointer.find_latest_experiment(out_folder)

        if last_exp is None:
            model.initialize_for_void_detection()
            print("Initialized model from scratch")
            return

        model_path = os.path.join(out_folder, last_exp, "last_model (copia).pth")

        # --- attempt 1: full load ---
        try:
            model.load_state_dict(torch.load(model_path, map_location="cuda"))
            print(f"Loaded full model from {last_exp}")
            return
        except (OSError, RuntimeError, KeyError):
            pass

        # --- attempt 2: partial load ---
        try:
            state = torch.load(model_path, map_location="cuda")
            model.load_state_dict(state, strict=False)
            not_loaded = sorted(set(model.state_dict().keys()) - set(state.keys()))
            print(f"Loaded partial weights from {last_exp}")
            print(f"  New layers kept from init: {len(not_loaded)} entries")
            for k in not_loaded:
                print(f"    {k}")
            return
        except (OSError, RuntimeError, KeyError):
            pass

        # --- attempt 3: fresh init ---
        model.initialize_for_void_detection()
        print("Initialized model from scratch (checkpoint not usable)")


# ──────────────────────────────────────────────
# Weighted Box Fusion (unchanged)
# ──────────────────────────────────────────────
def wbf(df, grid_s, box_fuser, cube_size=2000 / 16):
    """Apply Weighted Box Fusion to a DataFrame of voids."""
    if len(df) > 2:
        coords = torch.as_tensor(df[["x", "y", "z"]].values, dtype=torch.float32)
        distances = torch.cdist(coords, coords, p=2)
        scale = grid_s / cube_size
        labels = box_fuser.cluster(distances * scale)
        return group_means(df, labels)
    return df


# ──────────────────────────────────────────────
# Step 2: process_head
# ──────────────────────────────────────────────
def decode_and_fuse(tensor, sample_meta, config, box_fuser):
    """
    Decode all anchors from a raw tensor, apply WBF and overlap filtering.

    Works identically for both predictions and targets.

    Returns a filtered DataFrame of decoded voids.
    """
    tensor_cpu = tensor.detach().cpu()
    ox, oy, oz = sample_meta
    dfs = [
        wbf(
            target_to_voids(tensor_cpu[a * 5 : a * 5 + 5], a, ox, oy, oz, thresh=0.0),
            grid_s=tensor.shape[-1],
            box_fuser=box_fuser,
            cube_size=config.cube_size,
        )
        for a in range(config.n_anchors)
    ]
    return filter_touching_spheres(pd.concat(dfs, ignore_index=True))


def process_head(pred, target, sample_meta, config, box_fuser, loss_fn):
    """
    Evaluate a single detection head for one sample.

    Returns
    -------
    head_loss : Tensor | None
        Scalar loss on GPU (None when no ground-truth voids exist).
    metrics : dict
        All metrics needed by the logger.
    """
    clamped = torch.clamp(pred, min=-1.0, max=1.0) * (SCALE_ADJ - 0.5)

    df_pred = decode_and_fuse(clamped, sample_meta, config, box_fuser)
    df_target = decode_and_fuse(target, sample_meta, config, box_fuser)

    accuracy = compute_accuracy(clamped, target)

    loss_dict = None
    head_loss = None
    if len(df_target) > 0:
        loss_dict = loss_fn(clamped, target)
        head_loss = loss_dict["total_loss"]

    cent_err, rad_err, iou, prec, rec, _f1 = compute_cen_rad_iou_prec_rec_f1(
        df_pred, df_target
    )

    return head_loss, {
        "loss": loss_dict,
        "accuracy": accuracy,
        "rec": rec,
        "prec": prec,
        "iou": iou,
        "cent_err": cent_err,
        "rad_err": rad_err,
    }


def _log_head(logger, head, sample_id, metrics):
    """Push one head's results into the TrainingLogger."""
    loss = metrics["loss"]
    acc = metrics["accuracy"]

    if loss is not None:
        logger.det_update("total", head, sample_id, loss["total_loss"].item())
        logger.det_update("xyz", head, sample_id, loss["coordinate_loss"].item())
        logger.det_update("r", head, sample_id, loss["radius_loss"].item())
        logger.det_update("score", head, sample_id, loss["objectness_loss"].item())

    logger.det_update("obj", head, sample_id, acc["object_accuracy"].item())
    logger.det_update("no_obj", head, sample_id, acc["no_object_accuracy"].item())
    logger.det_update("acc", head, sample_id, acc["overall_accuracy"].item())
    logger.det_update("rec", head, sample_id, metrics["rec"])
    logger.det_update("prec", head, sample_id, metrics["prec"])
    logger.det_update("iou", head, sample_id, metrics["iou"])
    logger.det_update("cent_err", head, sample_id, metrics["cent_err"])
    logger.det_update("rad_err", head, sample_id, metrics["rad_err"])


# ──────────────────────────────────────────────
# Step 3: process_sample
# ──────────────────────────────────────────────
def process_sample(
    preds, targets_list, sample_id, batch, config, box_fuser, loss_fn, logger
):
    """
    Iterate over all heads for one sample inside a batch.

    Returns the accumulated (weighted) loss across heads.
    """
    n_heads = len(preds)
    sample_meta = (
        batch["ox"][sample_id].item(),
        batch["oy"][sample_id].item(),
        batch["oz"][sample_id].item(),
    )

    total_loss = 0
    for head in range(n_heads):
        pred = preds[head][sample_id]
        target = targets_list[head][sample_id]

        head_loss, metrics = process_head(
            pred,
            target,
            sample_meta,
            config,
            box_fuser,
            loss_fn,
        )
        _log_head(logger, head, sample_id, metrics)

        if head_loss is not None:
            total_loss = total_loss + head_loss * config.hwem**head

    return total_loss


# ──────────────────────────────────────────────
# Step 4: process_batch
# ──────────────────────────────────────────────
def process_batch(batch, model, loss_fn, config, box_fuser, logger, device):
    """
    Forward pass + per-sample loss accumulation for one batch.

    Returns the total (weighted) loss tensor.
    """
    with autocast(device):
        inputs = batch["input"].to(device)
        preds = model(inputs)
        targets_list = [t.to(device) for t in batch["targets"]]

        n_samples = len(preds[0])
        total_loss = 0
        for sid in range(n_samples):
            total_loss = total_loss + process_sample(
                preds,
                targets_list,
                sid,
                batch,
                config,
                box_fuser,
                loss_fn,
                logger,
            )
    return total_loss


def backward_step(total_loss, optimizer, gscaler, model, config):
    """Back-propagation with mixed-precision scaling and gradient clipping."""
    optimizer.zero_grad()
    gscaler.scale(total_loss / (config.hwem**4)).backward()
    gscaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    gscaler.step(optimizer)
    gscaler.update()


# ──────────────────────────────────────────────
# Step 5: run_epoch
# ──────────────────────────────────────────────
def run_epoch(
    epoch,
    mode,
    loader,
    model,
    loss_fn,
    optimizer,
    gscaler,
    config,
    box_fuser,
    logger,
    checkpointer,
    device,
):
    """Execute one full epoch (train or validation)."""
    logger.reset()
    if mode == "train":
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(mode == "train"):
        for batch in loader:
            total_loss = process_batch(
                batch,
                model,
                loss_fn,
                config,
                box_fuser,
                logger,
                device,
            )
            if mode == "train" and isinstance(total_loss, torch.Tensor):
                backward_step(total_loss, optimizer, gscaler, model, config)
            elif mode != "train" and isinstance(total_loss, torch.Tensor):
                checkpointer.save_if_best(model, total_loss)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.log_plt(mode)
    logger.print_all(epoch, mode)
    logger.update_all()
    logger.log(epoch, mode)


# ──────────────────────────────────────────────
# Main training loop (now slim)
# ──────────────────────────────────────────────
def train_detector(
    model, train_loader, valid_loader, config, exp_name="train_detector"
):
    """
    Main training loop.

    Args:
        model: network to train
        train_loader: training DataLoader
        valid_loader: validation DataLoader
        config: TrainingConfig with all hyper-parameters
        exp_name: experiment name (used for logging / checkpoints)
    """
    print("Start training")
    torch.autograd.set_detect_anomaly(True)

    loss_fn = YOLO3DLoss()
    optimizer = AdamW(
        params=list(model.parameters()) + list(loss_fn.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-06)
    gscaler = GradScaler()
    logger = TrainingLogger(exp_name, 5, n_samples=config.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device {device}")
    model.to(device)

    box_fuser = BoxClusterer()
    checkpointer = ModelCheckpointer(OUT_FOLDER, exp_name)
    loaders = {"train": train_loader, "val": valid_loader}

    # --- Profiler ---
    profiler = ResourceProfiler(
        out_dir=os.path.join(OUT_FOLDER, exp_name),
        interval=30,
    )
    profiler.start()

    # --- Graceful shutdown on SIGUSR1 (sent by SLURM before wall-time kill) ---
    # Note: SIGUSR1 is not available on Windows
    _shutdown = False

    def _handle_shutdown(signum, frame):
        """Signal handler to set shutdown flag when SIGUSR1 is received."""
        nonlocal _shutdown
        _shutdown = True
        print(
            f"\n[SIGNAL] Received signal {signum} in {frame} — will shut down."
        )

    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, _handle_shutdown)

    checkpointer.save_if_best(model, 0)
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()

        for mode in ["train", "val"]:
            profiler.set_epoch(epoch)
            profiler.set_note(f"{mode}_start")
            run_epoch(
                epoch,
                mode,
                loaders[mode],
                model,
                loss_fn,
                optimizer,
                gscaler,
                config,
                box_fuser,
                logger,
                checkpointer,
                device,
            )

        epoch_sec = time.perf_counter() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_sec:.1f}s")

        scheduler.step()
        logger.save_plots()
        logger.save_txt()
        checkpointer.save_last(model)

        # Check for graceful shutdown request
        if _shutdown:
            profiler.set_note("graceful_shutdown")
            print("[SHUTDOWN] Saving final checkpoint and flushing logs...")
            profiler.stop()
            print("[SHUTDOWN] Done. Exiting cleanly.")
            return

    profiler.stop()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def export_onnx(model, exp_name):
    """Optional ONNX export (kept separate from training)."""
    input_tensor = torch.zeros((1, Config3D.INPUT_CHANNELS, *Config3D.INPUT_SIZE))
    model.eval()
    model.to("cpu").float()
    input_tensor = input_tensor.float()
    torch.onnx.export(
        model,
        input_tensor,
        os.path.join("..", exp_name, "model.onnx"),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[f"out_{s}" for s in Config3D.SCALES],
    )
    print("Saved model ONNX")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")


def start_all(exp_name="voids_3d_detection"):
    """Main entry point to start training."""
    model = CosmicVoidDetectionVNet()

    ModelCheckpointer.load_or_init(model, OUT_FOLDER)

    config = TrainingConfig(batch_size=4, epochs=200)
    train_loader, valid_loader = void_data_loader(batch_size=config.batch_size)

    train_detector(model, train_loader, valid_loader, config=config, exp_name=exp_name)
