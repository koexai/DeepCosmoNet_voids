"""
this module aims to handle all the logging of the metrics
and the plot of the training curves for the project
"""
import os
import numpy as np
from src.logger_matplotlib import LivePlot
import matplotlib.pyplot as plt

from datetime import datetime
import json
from src.dcn_config import get_output_paths


OUT_FOLDER = get_output_paths()


class AverageValueMeter:
    """Meter to keep track of average values during training,
    such as losses and metrics."""

    def __init__(self):
        """Initialize the meter."""
        self.val = 0
        self.avg = np.nan
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset the meter to initial state."""
        self.val = 0
        self.avg = np.nan
        self.sum = 0
        self.count = 0

    def add(self, value, n=1):
        """Add a new value to the meter."""
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else np.nan

    def value(self):
        """Return the average and current value."""
        return self.avg, self.val


class TrainingLogger:
    """
    This class handles all the log of the metrics during the training
    """

    def __init__(self, exp_name, n_heads=5, n_samples=16):
        """Initialize the logger with experiment name, number of heads, and samples."""
        self.exp_name = exp_name
        self.n_heads = n_heads
        self.n_samples = n_samples
        # Creates directory for plots
        self.plot_dir = os.path.join(OUT_FOLDER, exp_name, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        # Creates directory for logs
        self.log_dir = os.path.join(OUT_FOLDER, exp_name, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses = {
            "total": "Loss total",
            "score": "Loss objectness",
            "xyz": "Loss centroid",
            "r": "Loss radius",
        }
        self.metrics = {
            "acc": "Accuracy",
            "obj": "Accuracy object",
            "no_obj": "Accuracy no object",
            "rec": "Recall",
            "prec": "Precision",
            "iou": "Average IoU",
            "cent_err": "Centroid Error",
            "rad_err": "Radius Error",
        }
        self.meters = {key: AverageValueMeter() for key in self.all_keys()}
        self.detailed_meters = {
            key: np.zeros((n_heads, n_samples)) for key in self.all_keys()
        }

        self.plt_loggers = {
            key: LivePlot(title) for key, title in self.all_titles().items()
        }
        # history for plots
        self.history = {key: {"train": [], "val": []} for key in self.all_keys()}
        self.epochs = []

        # log files
        self.log_file = os.path.join(self.log_dir, f"{exp_name}_log.txt")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Training Log - {exp_name}\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

    def all_keys(self):
        """
        Returns: all the metric keys short description
        """
        return list(self.losses.keys()) + list(self.metrics.keys())

    def all_titles(self):
        """
        Returns: all the metric extended description
        """
        return {**self.losses, **self.metrics}

    def reset(self):
        """
        Resets the logger
        """
        for meter in self.meters.values():
            meter.reset()
        for key in self.detailed_meters:
            self.detailed_meters[key] = self.detailed_meters[key] * np.nan

    def log(self, epoch, mode):
        """Log e crea grafici per l'epoca corrente"""

        # Prepara il messaggio di log
        log_message = f"\nEpoch {epoch} - {mode.upper()}\n"
        log_message += "-" * 60 + "\n"

        # Stampa e salva metriche
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch} - {mode.upper()}")
        print(f"{'=' * 60}")

        # Log losses
        print("\nLosses:")
        log_message += "Losses:\n"
        for key, title in self.losses.items():
            value = self.meters[key].value()[0]
            # Per le loss, mostra il valore diretto
            print(f"  {title:20s}: {value:.6f}")
            log_message += f"  {title:20s}: {value:.6f}\n"

            # Aggiungi alla storia
            self.history[key][mode].append(value)

        # Log metrics
        print("\nMetrics:")
        log_message += "\nMetrics:\n"
        for key, title in self.metrics.items():
            value = self.meters[key].value()[0]
            # Per le metriche, mostra come percentuale dove appropriato
            if key in ["acc", "obj", "no_obj", "rec", "prec", "iou"]:
                print(f"  {title:20s}: {value:.4f} ({value * 100:.2f}%)")
                log_message += f"  {title:20s}: {value:.4f} ({value * 100:.2f}%)\n"
            else:
                print(f"  {title:20s}: {value:.4f}")
                log_message += f"  {title:20s}: {value:.4f}\n"

            # Aggiungi alla storia
            self.history[key][mode].append(value)

        # Salva su file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message)

        # Aggiorna lista epoche solo quando completi train+val
        if mode == "val" and (len(self.epochs) == 0 or epoch > self.epochs[-1]):
            self.epochs.append(epoch)

        # Genera grafici dopo validation
        if mode == "val":
            self._plot_metrics(epoch)

            # Salva anche un file JSON con tutta la storia
            history_file = os.path.join(self.log_dir, f"{self.exp_name}_history.json")
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump({"epochs": self.epochs, "history": self.history}, f, indent=2)

    def _plot_metrics(self, current_epoch):
        """Crea grafici per tutte le metriche"""

        # Configurazione matplotlib
        plt.style.use("seaborn-v0_8-colorblind")
        plt.rcParams["figure.figsize"] = (12, 8)

        # Plot per le loss
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{self.exp_name} - Losses (Epoch {current_epoch})", fontsize=16)

        loss_keys = list(self.losses.keys())
        for idx, key in enumerate(loss_keys):
            ax = axes[idx // 2, idx % 2]

            # Plot train e validation
            if len(self.history[key]["train"]) > 0:
                ax.plot(
                    self.epochs[: len(self.history[key]["train"])],
                    self.history[key]["train"],
                    "b-",
                    label="Train",
                    linewidth=2,
                )
            if len(self.history[key]["val"]) > 0:
                ax.plot(
                    self.epochs[: len(self.history[key]["val"])],
                    self.history[key]["val"],
                    "r--",
                    label="Validation",
                    linewidth=2,
                )

            ax.set_title(self.losses[key], fontsize=14)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Log scale per le loss se appropriato
            if key in self.losses and len(self.history[key]["train"]) > 0:
                min_val = min(
                    min(self.history[key]["train"]),
                    min(self.history[key]["val"])
                    if self.history[key]["val"]
                    else float("inf"),
                )
                if min_val > 0:
                    ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plot_dir, "losses.png"), dpi=100, bbox_inches="tight"
        )
        plt.close()

        # Plot per le metriche - AGGIORNATO per 8 metriche
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f"{self.exp_name} - Metrics (Epoch {current_epoch})", fontsize=16)

        metric_keys = list(self.metrics.keys())
        for idx, key in enumerate(metric_keys):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            # Plot train e validation
            if len(self.history[key]["train"]) > 0:
                ax.plot(
                    self.epochs[: len(self.history[key]["train"])],
                    self.history[key]["train"],
                    "b-",
                    label="Train",
                    linewidth=2,
                )
            if len(self.history[key]["val"]) > 0:
                ax.plot(
                    self.epochs[: len(self.history[key]["val"])],
                    self.history[key]["val"],
                    "r--",
                    label="Validation",
                    linewidth=2,
                )

            ax.set_title(self.metrics[key], fontsize=14)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Y limits for percentage metrics
            if key in ["acc", "obj", "no_obj", "rec", "prec", "iou"]:
                ax.set_ylim([0, 1.05])
            # For centroid and radius errors, use automatic scale but start from 0
            elif key in ["cent_err", "rad_err"]:
                ax.set_ylim(bottom=0)

        # Hide empty cells (we have 8 metrics in a 3x3 grid = 9)
        for idx in range(len(metric_keys), 9):
            row, col = idx // 3, idx % 3
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plot_dir, "metrics.png"), dpi=100, bbox_inches="tight"
        )
        plt.close()

        # Plot combinato per una vista d'insieme - AGGIORNATO
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{self.exp_name} - Overview (Epoch {current_epoch})", fontsize=16)

        # Total loss
        if len(self.history["total"]["train"]) > 0:
            ax1.plot(
                self.epochs[: len(self.history["total"]["train"])],
                self.history["total"]["train"],
                "b-",
                label="Train",
                linewidth=2,
            )
            ax1.plot(
                self.epochs[: len(self.history["total"]["val"])],
                self.history["total"]["val"],
                "r--",
                label="Validation",
                linewidth=2,
            )
        ax1.set_title("Total Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if (
            len(self.history["total"]["train"]) > 0
            and min(self.history["total"]["train"]) > 0
        ):
            ax1.set_yscale("log")

        # Accuracy
        if len(self.history["acc"]["train"]) > 0:
            ax2.plot(
                self.epochs[: len(self.history["acc"]["train"])],
                self.history["acc"]["train"],
                "b-",
                label="Train",
                linewidth=2,
            )
            ax2.plot(
                self.epochs[: len(self.history["acc"]["val"])],
                self.history["acc"]["val"],
                "r--",
                label="Validation",
                linewidth=2,
            )
        ax2.set_title("Overall Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # IoU
        if len(self.history["iou"]["val"]) > 0:
            ax3.plot(
                self.epochs[: len(self.history["iou"]["val"])],
                self.history["iou"]["val"],
                "g-",
                label="Validation IoU",
                linewidth=2,
            )
        ax3.set_title("Average IoU")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("IoU")
        ax3.set_ylim([0, 1.05])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Errors (Centroid and Radius) - NEW
        if len(self.history["cent_err"]["val"]) > 0:
            ax4.plot(
                self.epochs[: len(self.history["cent_err"]["val"])],
                self.history["cent_err"]["val"],
                "orange",
                label="Centroid Error",
                linewidth=2,
            )
        if len(self.history["rad_err"]["val"]) > 0:
            ax4.plot(
                self.epochs[: len(self.history["rad_err"]["val"])],
                self.history["rad_err"]["val"],
                "purple",
                label="Radius Error",
                linewidth=2,
            )
        ax4.set_title("Detection Errors")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Error")
        ax4.set_ylim(bottom=0)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plot_dir, "overview.png"), dpi=100, bbox_inches="tight"
        )
        plt.close()

        print(f"\nPlots saved to {self.plot_dir}/")

    def log_plt(self, mode):
        """
        Calculates the global averages of the metrics and passes them to the plotter
        Args:
            mode: "train" or "val"
        """
        for key in self.all_keys():
            value = np.nanmean(self.detailed_meters[key])
            if value == np.nan:
                print(f"Warning: {key} = NaN")
                # value = 0
            log_val = np.log10(value) if key in self.losses else value
            self.plt_loggers[key].update(log_val, mode=mode)

    def update(self, key):
        """Updates the global average of a specific metric based on the detailed values
        Args:            key: the metric to update"""
        if key not in self.meters:
            raise KeyError(f"'{key}' is not a valid meter name.")
        value = np.nanmean(self.detailed_meters[key])
        if np.isnan(value):
            print(f"Warning: {key} = NaN")
            # value = 0
        self.meters[key].add(value, 1)

    def det_update(self, key, head, sample_id, value):
        """
        Updates a single value in the matrix of a specific metric
        Args:
            key: the metric to update
            head: the head where is measured
            sample_id: the id of the sample where is measured
            value: the value of the metric
        """
        if key not in self.detailed_meters:
            raise KeyError(f"'{key}' is not a valid meter name.")
        if head >= self.n_heads:
            raise KeyError(
                f"head {head} does not exist there is only {self.n_heads} heads."
            )
        if sample_id >= self.n_samples:
            raise KeyError(
                f"id {sample_id} is bigger than batch_size {self.n_samples}."
            )
        self.detailed_meters[key][head, sample_id] = value

    def save_plots(self):
        """
        Saves the training graphs
        """
        # self.saver.save()
        for key in self.all_keys():
            self.plt_loggers[key].save(self.plot_dir)

    def get_meter(self, key):
        """Returns the meter for a specific metric"""
        return self.meters.get(key)

    def save_txt(self):
        """Salva un riepilogo finale"""
        summary_file = os.path.join(self.log_dir, f"{self.exp_name}_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Training Summary - {self.exp_name}\n")
            f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("Final Results:\n")
            f.write("-" * 60 + "\n")

            # Risultati finali
            for key, title in self.all_titles().items():
                if len(self.history[key]["val"]) > 0:
                    final_val = self.history[key]["val"][-1]
                    best_val = (
                        min(self.history[key]["val"])
                        if key in self.losses
                        else max(self.history[key]["val"])
                    )
                    best_epoch = self.history[key]["val"].index(best_val) + 1

                    f.write(f"{title:20s}:\n")
                    f.write(f"  Final: {final_val:.6f}\n")
                    f.write(f"  Best:  {best_val:.6f} (epoch {best_epoch})\n\n")

        print(f"\nTraining completed. Summary saved to {summary_file}")
        print(f"All plots saved to {self.plot_dir}/")
        print(f"All logs saved to {self.log_dir}/")

    def print_all(self, epoch, mode):
        """
        Prints all the metrics
        Args:
            epoch: epoch number
            mode: "train" or "val"
        """
        print(mode, epoch)
        for key in self.detailed_meters:
            value = np.nanmean(self.detailed_meters[key])
            print(key, value)
        print()

    def update_all(self):
        """Updates all the global averages of the metrics
        based on the detailed values"""
        for key in self.meters:
            self.update(key)
