import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models import SmallCNN, get_resnet18
from evaluate import (collect_confidence_scores,
                      compute_distributional_overlap,
                      compute_deferral_performance)


# ─────────────────────────────────────────────
# ALL 19 CIFAR-10-C CORRUPTIONS
# ─────────────────────────────────────────────

CORRUPTIONS = [
    # Noise
    "gaussian_noise", "shot_noise", "impulse_noise",
    # Blur
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    # Weather
    "snow", "frost", "fog", "brightness",
    # Digital
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    # Extra
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]

SEVERITIES = [1, 2, 3, 4, 5]


# ─────────────────────────────────────────────
# 1. LOAD CIFAR-10-C
# ─────────────────────────────────────────────

def load_cifar10c(corruption, severity, data_dir="./data/CIFAR-10-C"):
    """
    Loads a specific corruption + severity from CIFAR-10-C.

    CIFAR-10-C structure:
      data_dir/
        gaussian_noise.npy   # shape (50000, 32, 32, 3) — all 5 severities stacked
        shot_noise.npy
        ...
        labels.npy           # shape (50000,) — same labels repeated 5 times

    Each severity has 10000 images:
      severity 1 → indices [0,     10000)
      severity 2 → indices [10000, 20000)
      ...
      severity 5 → indices [40000, 50000)

    Download from: https://zenodo.org/record/2535967
    """
    assert os.path.exists(data_dir), (
        f"CIFAR-10-C not found at {data_dir}.\n"
        f"Download from https://zenodo.org/record/2535967\n"
        f"and extract to {data_dir}"
    )

    images_path = os.path.join(data_dir, f"{corruption}.npy")
    labels_path = os.path.join(data_dir, "labels.npy")

    assert os.path.exists(images_path), \
        f"Corruption file not found: {images_path}"

    images = np.load(images_path)   # (50000, 32, 32, 3) uint8
    labels = np.load(labels_path)   # (50000,)

    # Slice out the correct severity block
    start = (severity - 1) * 10000
    end   =  severity      * 10000
    images = images[start:end]      # (10000, 32, 32, 3)
    labels = labels[start:end]      # (10000,)

    # Normalise to tensor — same stats as CIFAR-10 training
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])

    images = images.astype(np.float32) / 255.0                    # [0, 1]
    images = (images - mean).astype(np.float32) / std             # normalise
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    labels = torch.tensor(labels).long()

    dataset = TensorDataset(images, labels)
    loader  = DataLoader(dataset, batch_size=128,
                         shuffle=False, num_workers=2)
    return loader


# ─────────────────────────────────────────────
# 2. EVALUATE ONE MODEL ON ONE CORRUPTION
# ─────────────────────────────────────────────

def evaluate_corruption(model_s, model_l, corruption,
                         severity, data_dir, device="cpu"):
    """
    Returns s_o, s_d, acc_s for a given corruption + severity.
    """
    loader = load_cifar10c(corruption, severity, data_dir)

    correct_confs, incorrect_confs = collect_confidence_scores(
        model_s, loader, device)

    s_o, _, _, _ = compute_distributional_overlap(
        correct_confs, incorrect_confs)

    _, _, _, _, s_d, acc_s, _ = compute_deferral_performance(
        model_s, model_l, loader, device=device)

    return s_o, s_d, acc_s


# ─────────────────────────────────────────────
# 3. FULL ROBUSTNESS SWEEP
# All corruptions × all severities × all alphas
# ─────────────────────────────────────────────

def run_robustness_evaluation(alphas, data_dir="./data/CIFAR-10-C",
                               num_classes=10, device="cpu"):
    """
    Runs the full robustness evaluation:
      - For each alpha (+ baseline)
      - For each of 19 corruptions
      - For each of 5 severity levels
      → Records s_o, s_d, acc_s

    Returns:
        rob_results : nested dict
                      rob_results[key][corruption][severity]
                      = {"s_o": ..., "s_d": ..., "acc_s": ...}
    """
    keys  = ["baseline"] + alphas
    ckpts = {"baseline": "model_s_pretrained.pth"}
    ckpts.update({a: f"model_s_gk_alpha{a}.pth" for a in alphas})

    # Load ML once — shared across all evaluations
    model_l = get_resnet18(num_classes=num_classes).to(device)
    model_l.load_state_dict(
        torch.load("model_l_pretrained.pth", map_location=device))
    model_l.eval()

    rob_results = {}

    for key in keys:
        label = "baseline" if key == "baseline" else f"alpha={key}"
        print(f"\n{'='*50}")
        print(f"Evaluating: {label}")
        print(f"{'='*50}")

        model_s = SmallCNN(num_classes=num_classes).to(device)
        model_s.load_state_dict(
            torch.load(ckpts[key], map_location=device))
        model_s.eval()

        rob_results[key] = {}

        for corruption in CORRUPTIONS:
            rob_results[key][corruption] = {}
            for severity in SEVERITIES:
                s_o, s_d, acc_s = evaluate_corruption(
                    model_s, model_l, corruption,
                    severity, data_dir, device)

                rob_results[key][corruption][severity] = {
                    "s_o":   s_o,
                    "s_d":   s_d,
                    "acc_s": acc_s,
                }
                print(f"  {corruption:<25} sev={severity}  "
                      f"s_o={s_o:.4f}  s_d={s_d:.4f}  acc_s={acc_s:.4f}")

    return rob_results


# ─────────────────────────────────────────────
# 4. PLOT: Mean Corruption Error (mCE) style
#    acc_s averaged across all corruptions
#    per severity — for each alpha
# ─────────────────────────────────────────────

def plot_robustness_vs_severity(rob_results, alphas,
                                 save_path="plot_robustness_severity.png"):
    """
    Three subplots (s_o, s_d, acc_s) × x-axis = severity level.
    Each line = one alpha value.
    Shows how metrics degrade as severity increases.
    """
    keys   = ["baseline"] + alphas
    labels = ["Baseline"] + [f"α={a}" for a in alphas]
    colors = ["#E8855A", "#C6DBEF", "#9ECAE1",
              "#6BAED6", "#2171B5", "#08306B"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics   = ["s_o", "s_d", "acc_s"]
    titles    = ["Distributional Overlap $s_o$ ↓",
                 "Deferral Performance $s_d$ ↑",
                 "Small Model Accuracy $acc(M_S)$ ↑"]

    for ax, metric, title in zip(axes, metrics, titles):
        for key, label, color in zip(keys, labels, colors):
            # Average across all 19 corruptions per severity
            mean_per_severity = []
            for sev in SEVERITIES:
                vals = [rob_results[key][c][sev][metric]
                        for c in CORRUPTIONS]
                mean_per_severity.append(np.mean(vals))

            ls = "--" if key == "baseline" else "-"
            ax.plot(SEVERITIES, mean_per_severity,
                    marker='o', label=label,
                    color=color, linestyle=ls, linewidth=2)

        ax.set_xlabel("Corruption Severity", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(SEVERITIES)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness Under Natural Corruptions — CIFAR-10-C",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# 5. PLOT: Heatmap — acc_s per corruption × severity
#    for a chosen alpha vs baseline
# ─────────────────────────────────────────────

def plot_corruption_heatmap(rob_results, key="baseline",
                             metric="acc_s",
                             save_path="plot_heatmap.png"):
    """
    Heatmap of a chosen metric across all corruptions (rows)
    and severities (columns) for a given alpha.
    Useful for spotting which corruptions hurt most.
    """
    data = np.zeros((len(CORRUPTIONS), len(SEVERITIES)))
    for i, c in enumerate(CORRUPTIONS):
        for j, s in enumerate(SEVERITIES):
            data[i, j] = rob_results[key][c][s][metric]

    label = "Baseline" if key == "baseline" else f"α={key}"
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(data, aspect='auto',
                   cmap='RdYlGn' if metric != "s_o" else 'RdYlGn_r')

    ax.set_xticks(range(len(SEVERITIES)))
    ax.set_xticklabels([f"Sev {s}" for s in SEVERITIES], fontsize=10)
    ax.set_yticks(range(len(CORRUPTIONS)))
    ax.set_yticklabels(CORRUPTIONS, fontsize=9)
    ax.set_title(f"{metric} — {label}", fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    alphas    = [0.9, 0.7, 0.5, 0.3, 0.1]
    data_dir  = "./data/CIFAR-10-C"

    # ── Step 1: Run full robustness evaluation ──
    rob_results = run_robustness_evaluation(
        alphas, data_dir=data_dir, device=device)

    # Save results
    with open("rob_results.pkl", "wb") as f:
        pickle.dump(rob_results, f)
    print("\nSaved → rob_results.pkl")

    # ── Step 2: Plot metrics vs severity ──
    plot_robustness_vs_severity(
        rob_results, alphas,
        save_path="plot_robustness_severity.png")

    # ── Step 3: Heatmaps for baseline and best alpha ──
    plot_corruption_heatmap(
        rob_results, key="baseline", metric="acc_s",
        save_path="plot_heatmap_baseline.png")

    plot_corruption_heatmap(
        rob_results, key=0.1, metric="acc_s",
        save_path="plot_heatmap_alpha01.png")
