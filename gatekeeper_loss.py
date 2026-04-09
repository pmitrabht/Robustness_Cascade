import torch
import torch.nn as nn
import torch.nn.functional as F


class GatekeeperLoss(nn.Module):
    """
    Gatekeeper Loss — Rabanser et al., 2025
    ----------------------------------------
    L = alpha * L_corr + (1 - alpha) * L_incorr

    L_corr   : Cross-entropy on CORRECTLY predicted samples
               → encourages MS to be MORE confident when right

    L_incorr : KL divergence to Uniform on INCORRECTLY predicted samples
               → encourages MS to spread probability evenly when wrong
               → low confidence = high entropy = "please defer me"

    Args:
        alpha (float): Trade-off parameter in (0, 1)
            - Low alpha  (e.g. 0.1) → aggressive deferral, lower MS accuracy
            - High alpha (e.g. 0.9) → better MS accuracy, weaker deferral
        num_classes (int): Number of output classes
    """

    def __init__(self, alpha: float = 0.5, num_classes: int = 10):
        super().__init__()
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, C) raw model outputs from MS
            labels : (B,)  ground truth class indices
        Returns:
            scalar loss
        """
        preds = logits.argmax(dim=-1)

        correct_mask   = (preds == labels)   # samples MS got right
        incorrect_mask = ~correct_mask        # samples MS got wrong

        # ── L_corr: cross-entropy on correct predictions ──
        if correct_mask.sum() > 0:
            L_corr = F.cross_entropy(
                logits[correct_mask],
                labels[correct_mask]
            )
        else:
            L_corr = torch.tensor(0.0, device=logits.device)

        # ── L_incorr: KL(p_i || Uniform) on incorrect predictions ──
        if incorrect_mask.sum() > 0:
            uniform = torch.full(
                (incorrect_mask.sum(), self.num_classes),
                fill_value=1.0 / self.num_classes,
                device=logits.device
            )
            L_incorr = F.kl_div(
                F.log_softmax(logits[incorrect_mask], dim=-1),
                uniform,
                reduction='batchmean'
            )
        else:
            L_incorr = torch.tensor(0.0, device=logits.device)

        return self.alpha * L_corr + (1 - self.alpha) * L_incorr
