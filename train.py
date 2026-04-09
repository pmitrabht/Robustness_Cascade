import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import SmallCNN, get_resnet18
from gatekeeper_loss import GatekeeperLoss


# ─────────────────────────────────────────────
# STAGE 1 — Standard Training (Cross-Entropy)
# Used for both MS and ML independently
# ─────────────────────────────────────────────

def train_standard(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   epochs: int = 30,
                   lr: float = 1e-3,
                   device: str = "cpu") -> nn.Module:
    """
    Standard training with cross-entropy loss.
    Run this for BOTH SmallCNN (MS) and ResNet-18 (ML)
    before any Gatekeeper fine-tuning.

    Args:
        model        : SmallCNN or ResNet-18
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        epochs       : number of epochs
        lr           : learning rate
        device       : 'cpu' or 'cuda'
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):

        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # ── Validate ──
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch+1:02d}/{epochs}] "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_correct/train_total:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    return model


# ─────────────────────────────────────────────
# STAGE 2 — Gatekeeper Fine-tuning (MS only)
# ML is frozen — only MS gets updated
# ─────────────────────────────────────────────

def finetune_gatekeeper(model_s: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         alpha: float = 0.5,
                         num_classes: int = 10,
                         epochs: int = 10,
                         lr: float = 1e-4,
                         device: str = "cpu") -> nn.Module:
    """
    Fine-tune MS with GatekeeperLoss.
    MS should already be pre-trained (Stage 1) before calling this.
    ML is not touched here at all.

    Args:
        model_s      : pre-trained SmallCNN (MS)
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        alpha        : Gatekeeper trade-off parameter (0 < alpha < 1)
                       paper sweeps: [0.1, 0.3, 0.5, 0.7, 0.9]
        num_classes  : number of output classes
        epochs       : fine-tuning epochs (fewer than Stage 1)
        lr           : lower lr than Stage 1 since we're fine-tuning
        device       : 'cpu' or 'cuda'
    """
    model_s = model_s.to(device)
    criterion = GatekeeperLoss(alpha=alpha, num_classes=num_classes)
    optimizer = optim.Adam(model_s.parameters(), lr=lr)

    print(f"\n── Gatekeeper Fine-tuning | alpha={alpha} ──")

    for epoch in range(epochs):

        # ── Fine-tune MS ──
        model_s.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model_s(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ── Validate MS accuracy after fine-tuning ──
        val_acc = evaluate(model_s, val_loader, device)

        print(f"Epoch [{epoch+1:02d}/{epochs}] "
              f"GK Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc (MS): {val_acc:.4f}")

    return model_s


# ─────────────────────────────────────────────
# HELPER — Accuracy Evaluation
# ─────────────────────────────────────────────

def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str = "cpu") -> float:
    """Returns accuracy of model on the given loader."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset: CIFAR-10 ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=2)

    # ── Stage 1: Train MS and ML independently ──
    model_s = SmallCNN(num_classes=10)
    model_l = get_resnet18(num_classes=10)

    print("=== Stage 1: Training MS (SmallCNN) ===")
    model_s = train_standard(model_s, train_loader, val_loader,
                              epochs=30, lr=1e-3, device=device)

    print("\n=== Stage 1: Training ML (ResNet-18) ===")
    model_l = train_standard(model_l, train_loader, val_loader,
                              epochs=30, lr=1e-3, device=device)

    # Save Stage 1 checkpoints
    torch.save(model_s.state_dict(), "model_s_pretrained.pth")
    torch.save(model_l.state_dict(), "model_l_pretrained.pth")

    # ── Stage 2: Fine-tune MS with Gatekeeper loss ──
    # Paper sweeps alpha across [0.1, 0.3, 0.5, 0.7, 0.9]
    print("\n=== Stage 2: Gatekeeper Fine-tuning ===")
    for alpha in [0.9, 0.7, 0.5, 0.3, 0.1]:
        # Reload fresh Stage 1 MS for each alpha run
        model_s_ft = SmallCNN(num_classes=10)
        model_s_ft.load_state_dict(torch.load("model_s_pretrained.pth"))

        model_s_ft = finetune_gatekeeper(
            model_s_ft, train_loader, val_loader,
            alpha=alpha, num_classes=10,
            epochs=10, lr=1e-4, device=device
        )

        torch.save(model_s_ft.state_dict(), f"model_s_gk_alpha{alpha}.pth")
        print(f"Saved → model_s_gk_alpha{alpha}.pth")
