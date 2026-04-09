import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
# PLOT 1: Distributional Overlap (Figure 3a)
# ─────────────────────────────────────────────

def plot_distributional_overlap(results, alpha_to_show=0.1,
                                  save_path="plot_overlap.png"):
    """
    Reproduces Figure 3a from the paper.
    Shows KDE of confidence scores for correct vs incorrect predictions,
    with the overlap area shaded.

    alpha_to_show: which alpha result to visualise (default: 0.1 = most aggressive)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    keys    = ["baseline", alpha_to_show]
    titles  = ["Baseline (no Gatekeeper)", f"Gatekeeper α={alpha_to_show}"]
    colors  = {"correct": "#4878CF", "incorrect": "#E8855A", "overlap": "#C97FB3"}

    for ax, key, title in zip(axes, keys, titles):
        r    = results[key]
        grid = r["grid"]
        dc   = r["dens_c"]
        di   = r["dens_i"]

        ax.plot(grid, dc, color=colors["correct"],   lw=2, label="Correct")
        ax.plot(grid, di, color=colors["incorrect"], lw=2, label="Incorrect")
        ax.fill_between(grid, np.minimum(dc, di),
                         alpha=0.4, color=colors["overlap"],
                         label=f"Overlap area $s_o$={r['s_o']:.3f}")

        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Density",    fontsize=12)
        ax.set_title(title,         fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("a) Distributional Overlap $s_o$", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# PLOT 2: Deferral Performance (Figure 3b)
# ─────────────────────────────────────────────

def plot_deferral_performance(results, alpha_to_show=0.1,
                               save_path="plot_deferral.png"):
    """
    Reproduces Figure 3b from the paper.
    Shows ideal, random, and realized deferral curves
    with performance and headroom areas shaded.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    keys   = ["baseline", alpha_to_show]
    titles = ["Baseline (no Gatekeeper)", f"Gatekeeper α={alpha_to_show}"]

    for ax, key, title in zip(axes, keys, titles):
        r  = results[key]
        dr = r["deferral_ratios"]
        ar = r["accs_real"]
        an = r["accs_rand"]
        ai = r["accs_ideal"]

        # ── Shaded areas ──
        # Blue: performance area (realized - random)
        ax.fill_between(dr, an, ar,
                         where=(ar >= an),
                         alpha=0.4, color="#6EB5E0",
                         label="Performance Area ($A_{perf}$)")

        # Green hatched: headroom (ideal - realized)
        ax.fill_between(dr, ar, ai,
                         where=(ai >= ar),
                         alpha=0.25, color="#5DBB63", hatch='///',
                         label="Headroom Area")

        # ── Curves ──
        ax.plot(dr, ai, color="#2CA02C", lw=2,
                linestyle='--', label="Ideal Deferral ($acc_{ideal}$)")
        ax.plot(dr, an, color="#D62728", lw=2,
                linestyle=':', label="Random Deferral ($acc_{rand}$)")
        ax.plot(dr, ar, color="black",   lw=2.5,
                label=f"Realized Deferral ($s_d$={r['s_d']:.3f})")

        # ── Dots: no deferral and full deferral ──
        ax.scatter([0], [r["acc_s"]], color="#BCBD22",
                   zorder=5, s=80, label=f"No deferral acc($M_S$)={r['acc_s']:.3f}")
        ax.scatter([1], [r["acc_l"]], color="#9467BD",
                   zorder=5, s=80, marker='s',
                   label=f"Full deferral acc($M_L$)={r['acc_l']:.3f}")

        ax.set_xlabel("Deferral Ratio",  fontsize=12)
        ax.set_ylabel("Joint Accuracy",  fontsize=12)
        ax.set_title(title,              fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("b) Deferral Performance $s_d$", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# PLOT 3: Alpha Sweep Summary (like Figure 4)
# Shows s_o, s_d, acc_s across all alpha values
# ─────────────────────────────────────────────

def plot_alpha_sweep(results, alphas,
                     save_path="plot_alpha_sweep.png"):
    """
    Reproduces Figure 4 style: three subplots showing how
    s_o, s_d, and acc_s change as alpha varies.
    """
    keys   = ["baseline"] + alphas
    labels = ["Base"] + [str(a) for a in alphas]

    s_o_vals   = [results[k]["s_o"]   for k in keys]
    s_d_vals   = [results[k]["s_d"]   for k in keys]
    acc_s_vals = [results[k]["acc_s"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bar_color = "#5B9BD5"

    # s_o — lower is better
    axes[0].bar(labels, s_o_vals, color=bar_color, edgecolor='black', width=0.5)
    axes[0].set_title("Distributional Overlap $s_o$ ↓", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("$s_o$")
    axes[0].grid(axis='y', alpha=0.3)

    # s_d — higher is better
    axes[1].bar(labels, s_d_vals, color="#5DBB63", edgecolor='black', width=0.5)
    axes[1].set_title("Deferral Performance $s_d$ ↑", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("$s_d$")
    axes[1].grid(axis='y', alpha=0.3)

    # acc_s — context (not directly better/worse)
    axes[2].bar(labels, acc_s_vals, color="#E8855A", edgecolor='black', width=0.5)
    axes[2].set_title("Small Model Accuracy $acc(M_S)$", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Alpha")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle("CIFAR-10 — Gatekeeper Alpha Sweep", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    with open("eval_results.pkl", "rb") as f:
        results = pickle.load(f)

    alphas = [0.9, 0.7, 0.5, 0.3, 0.1]

    # Figure 3a style — distributional overlap
    plot_distributional_overlap(results, alpha_to_show=0.1,
                                 save_path="plot_overlap.png")

    # Figure 3b style — deferral performance
    plot_deferral_performance(results, alpha_to_show=0.1,
                               save_path="plot_deferral.png")

    # Figure 4 style — alpha sweep summary
    plot_alpha_sweep(results, alphas,
                     save_path="plot_alpha_sweep.png")
