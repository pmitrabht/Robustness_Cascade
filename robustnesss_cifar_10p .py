import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import SmallCNN, get_resnet18


# ============================================================
# CONFIG
# ============================================================
# Full list if you want later:
# PERTURBATIONS = [
#     "gaussian_noise", "shot_noise", "motion_blur", "zoom_blur",
#     "spatter", "brightness", "translate", "rotate", "tilt", "scale",
#     "gaussian_blur", "speckle_noise", "snow", "shear",
# ]

PERTURBATIONS = ["gaussian_noise", "motion_blur", "brightness", "rotate"]
N_IMAGES = 10000


# ============================================================
# 0. DEVICE HELPER
# ============================================================
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


# ============================================================
# 1. LOAD CIFAR-10-P
# ============================================================
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_cifar10p(perturbation, data_dir="./data/CIFAR-10-P", batch_size=128):
    """
    Returns:
        loaders : list of DataLoaders, one per perturbation step
        n_steps : number of perturbation steps

    Supports:
      - flat format:                (n_steps * 10000, 32, 32, 3)
      - stacked format (step first):(n_steps, 10000, 32, 32, 3)
      - stacked format (img first): (10000, n_steps, 32, 32, 3)   <-- YOUR CASE
    """
    assert os.path.exists(data_dir), (
        f"CIFAR-10-P not found at {data_dir}.\n"
        f"Download from https://zenodo.org/record/2535967"
    )

    images_path = os.path.join(data_dir, f"{perturbation}.npy")
    assert os.path.exists(images_path), f"Perturbation file not found: {images_path}"

    images = np.load(images_path)
    print(f"[DEBUG] Loaded {perturbation}: shape={images.shape}, dtype={images.dtype}")

    n_images = 10000  # CIFAR-10 test set size

    # --------------------------------------------------------
    # Normalize shape -> (n_steps, 10000, 32, 32, 3)
    # --------------------------------------------------------
    if images.ndim == 4:
        # flat: (n_steps * 10000, 32, 32, 3)
        assert images.shape[0] % n_images == 0, (
            f"Flat CIFAR-10-P shape {images.shape[0]} is not divisible by {n_images}"
        )
        n_steps = images.shape[0] // n_images
        images = images.reshape(n_steps, n_images, 32, 32, 3)

    elif images.ndim == 5:
        # Case 1: already (n_steps, 10000, 32, 32, 3)
        if images.shape[1] == n_images:
            n_steps = images.shape[0]

        # Case 2: (10000, n_steps, 32, 32, 3)  <-- YOUR FILES
        elif images.shape[0] == n_images:
            n_steps = images.shape[1]
            images = np.transpose(images, (1, 0, 2, 3, 4))  # -> (n_steps, 10000, 32, 32, 3)

        else:
            raise ValueError(f"Unexpected 5D CIFAR-10-P shape: {images.shape}")

    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    print(f"[DEBUG] Interpreted as n_steps={n_steps}, n_images={n_images}")
    print(f"[DEBUG] Final normalized shape: {images.shape}")

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    labels = None
    for label_fname in ["labels.npy", "cifar10_labels.npy", "labels_hard.npy", "label.npy"]:
        label_path = os.path.join(data_dir, label_fname)
        if os.path.exists(label_path):
            labels = np.load(label_path)
            print(f"[DEBUG] Loaded labels from {label_fname}, shape={labels.shape}")
            break

    if labels is None:
        import torchvision
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
        labels = np.array(test_set.targets)
        print("[DEBUG] Labels loaded from torchvision CIFAR-10 test set")

    labels = labels[:n_images]
    assert len(labels) == n_images, f"Expected 10000 labels, got {len(labels)}"

    # --------------------------------------------------------
    # Normalization (same as CIFAR training)
    # --------------------------------------------------------
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    pin_memory = torch.cuda.is_available()

    loaders = []
    for step in range(n_steps):
        imgs = images[step]  # (10000, 32, 32, 3)

        assert imgs.shape == (n_images, 32, 32, 3), f"Unexpected step shape: {imgs.shape}"

        imgs = imgs.astype(np.float32) / 255.0
        imgs = ((imgs - mean) / std).astype(np.float32)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()

        lbls = torch.from_numpy(labels.astype(np.int64))

        loader = DataLoader(
            TensorDataset(imgs, lbls),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory
        )
        loaders.append(loader)

    return loaders, n_steps


# ============================================================
# 2. COMPUTE FLIP RATES PER STEP (MS + DEFERRAL + CASCADE)
# ============================================================
def compute_flip_rates(model_s, model_l, loaders, tau=0.7, device=None):
    """
    Computes per-step outputs and consecutive-step flip rates.

    Returns:
        pred_flip_rates_ms      : MS prediction flip rate between steps
        deferral_flip_rates     : deferral decision flip rate between steps
        pred_flip_rates_cascade : FINAL cascade prediction flip rate between steps
        acc_ms_per_step         : MS accuracy at each step
        acc_cascade_per_step    : final cascade accuracy at each step
        defer_rate_per_step     : fraction deferred at each step
    """
    if device is None:
        device = get_device()

    model_s = model_s.to(device)
    model_l = model_l.to(device)
    model_s.eval()
    model_l.eval()

    all_preds_ms = []
    all_preds_cascade = []
    all_deferred = []
    all_acc_ms = []
    all_acc_cascade = []
    all_defer_rates = []

    for loader in loaders:
        preds_ms = []
        preds_cascade = []
        deferred = []
        correct_ms = []
        correct_cascade = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Small model
                logits_s = model_s(images)
                probs_s = F.softmax(logits_s, dim=-1)
                conf_s, pred_s = probs_s.max(dim=-1)

                # Large model
                logits_l = model_l(images)
                pred_l = logits_l.argmax(dim=-1)

                # Deferral decision
                defer_mask = conf_s < tau

                # Final cascade prediction
                pred_final = pred_s.clone()
                pred_final[defer_mask] = pred_l[defer_mask]

                # Store
                preds_ms.extend(pred_s.cpu().numpy())
                preds_cascade.extend(pred_final.cpu().numpy())
                deferred.extend(defer_mask.cpu().numpy())

                correct_ms.extend((pred_s == labels).cpu().numpy())
                correct_cascade.extend((pred_final == labels).cpu().numpy())

        preds_ms = np.array(preds_ms)
        preds_cascade = np.array(preds_cascade)
        deferred = np.array(deferred)

        all_preds_ms.append(preds_ms)
        all_preds_cascade.append(preds_cascade)
        all_deferred.append(deferred)
        all_acc_ms.append(np.mean(correct_ms))
        all_acc_cascade.append(np.mean(correct_cascade))
        all_defer_rates.append(np.mean(deferred))

    # Consecutive-step flip rates
    pred_flips_ms = []
    defer_flips = []
    pred_flips_cascade = []

    n_steps = len(loaders)
    for t in range(n_steps - 1):
        pred_flips_ms.append((all_preds_ms[t] != all_preds_ms[t + 1]).mean())
        defer_flips.append((all_deferred[t] != all_deferred[t + 1]).mean())
        pred_flips_cascade.append((all_preds_cascade[t] != all_preds_cascade[t + 1]).mean())

    return (
        np.array(pred_flips_ms),
        np.array(defer_flips),
        np.array(pred_flips_cascade),
        np.array(all_acc_ms),
        np.array(all_acc_cascade),
        np.array(all_defer_rates),
    )


# ============================================================
# 3. FULL SWEEP (ONE PERTURBATION AT A TIME)
# ============================================================
import os
import pickle
import numpy as np
import torch

# =========================================================
# Assumes these already exist in your notebook/script:
# - get_device()
# - SmallCNN
# - get_resnet18()
# - load_cifar10p()
# - compute_flip_rates()
# =========================================================


def run_cifar10p_evaluation(
    alphas,
    perturbations,
    data_dir="./data/CIFAR-10-P",
    tau=0.7,
    num_classes=10,
    device=None,
    output_dir="."
):
    """
    Runs CIFAR-10-P evaluation for:
      - baseline
      - all alpha values
    for the given list of perturbations (recommended: one perturbation at a time).

    Saves partial results incrementally after:
      - baseline
      - each alpha

    Partial files are saved inside output_dir.
    """

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    if device is None:
        device = get_device()

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------
    keys = ["baseline"] + alphas
    ckpts = {"baseline": "model_s_pretrained.pth"}
    ckpts.update({a: f"model_s_gk_alpha{a}.pth" for a in alphas})

    # --------------------------------------------------------
    # Load large model ONCE
    # --------------------------------------------------------
    print("\n[INFO] Loading large model...")
    model_l = get_resnet18(num_classes=num_classes).to(device)
    model_l.load_state_dict(torch.load("model_l_pretrained.pth", map_location=device))
    model_l.eval()
    print("[INFO] Large model loaded.")

    # --------------------------------------------------------
    # Result container
    # --------------------------------------------------------
    p_results = {}

    # --------------------------------------------------------
    # Loop over perturbations
    # --------------------------------------------------------
    for pert in perturbations:
        print(f"\n{'='*60}")
        print(f"[INFO] Running perturbation: {pert}")
        print(f"{'='*60}")

        p_results[pert] = {}

        # partial save file for this perturbation
        partial_name = os.path.join(output_dir, f"cifar10p_results_partial_{pert}.pkl")

        # Load CIFAR-10-P loaders ONCE per perturbation
        print(f"[INFO] Loading CIFAR-10-P data for: {pert}")
        loaders, n_steps = load_cifar10p(pert, data_dir)
        print(f"[INFO] Loaded {n_steps} perturbation steps.")

        # ----------------------------------------------------
        # Loop over baseline + alpha values
        # ----------------------------------------------------
        for key in keys:
            label = "Baseline" if key == "baseline" else f"α={key}"
            print(f"\n--- Evaluating {label} on {pert} ---")

            # Load small model
            model_s = SmallCNN(num_classes=num_classes).to(device)
            model_s.load_state_dict(torch.load(ckpts[key], map_location=device))
            model_s.eval()

            # Compute flip rates / metrics
            (
                pfr_ms,
                dfr,
                pfr_cascade,
                acc_ms,
                acc_cascade,
                defer_rates,
            ) = compute_flip_rates(
                model_s, model_l, loaders, tau=tau, device=device
            )

            # Aggregate
            mfp_ms = float(np.mean(pfr_ms))
            mfp_cascade = float(np.mean(pfr_cascade))
            mean_dfr = float(np.mean(dfr))

            # Save under this perturbation
            p_results[pert][key] = {
                "pred_flip_rates_ms": pfr_ms,
                "deferral_flip_rates": dfr,
                "pred_flip_rates_cascade": pfr_cascade,
                "acc_ms_per_step": acc_ms,
                "acc_cascade_per_step": acc_cascade,
                "defer_rate_per_step": defer_rates,
                "mfp_ms": mfp_ms,
                "mfp_cascade": mfp_cascade,
                "mean_deferral_flip": mean_dfr,
                "n_steps": n_steps,
            }

            # ------------------------------------------------
            # SAVE PARTIAL AFTER EACH MODEL
            # (baseline + each alpha)
            # ------------------------------------------------
            with open(partial_name, "wb") as f:
                pickle.dump(p_results[pert], f)

            print(f"[INFO] Partial saved → {partial_name}")
            print(
                f"{label:>10} | "
                f"mFP-MS = {mfp_ms:.4f} | "
                f"mFP-Cascade = {mfp_cascade:.4f} | "
                f"Deferral Flip = {mean_dfr:.4f}"
            )

            # cleanup
            del model_s
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return p_results


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    device = get_device()
    alphas = [0.9, 0.7, 0.5, 0.3, 0.1]
    data_dir = "./data/CIFAR-10-P"
    tau = 0.7

    print("\n[INFO] Current working directory:", os.getcwd())

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY HERE
    # --------------------------------------------------------
    output_dir = os.path.join(os.getcwd(), "cifar10p_outputs")
    os.makedirs(output_dir, exist_ok=True)
    print("[INFO] Output directory:", output_dir)
    
    loaders, n_steps = load_cifar10p("gaussian_noise", "./data/CIFAR-10-P")
    print("n_steps:", n_steps)

    x, y = next(iter(loaders[0]))
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("first 10 labels:", y[:10].tolist())
    print("x min/max:", x.min().item(), x.max().item())

    # --------------------------------------------------------
    # CHANGE THIS EACH RUN → run ONE perturbation at a time
    # --------------------------------------------------------
    #perturbations = ["gaussian_noise"]
    # Examples:
    #perturbations = ["motion_blur"]
    #perturbations = ["brightness"]
    perturbations = ["rotate"]

    p_results = run_cifar10p_evaluation(
        alphas=alphas,
        perturbations=perturbations,
        data_dir=data_dir,
        tau=tau,
        num_classes=10,
        device=device,
        output_dir=output_dir
    )

    save_tag = perturbations[0]
    final_name = os.path.join(output_dir, f"cifar10p_results_{save_tag}.pkl")

    with open(final_name, "wb") as f:
        pickle.dump(p_results, f)

    print(f"\n[INFO] Final saved → {final_name}")
    print("[INFO] Done.")