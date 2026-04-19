import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import SmallCNN, get_resnet18
from gatekeeper_loss import GatekeeperLoss


# ─────────────────────────────────────────────
# STAGE 1a — Train MS (SmallCNN)
# Adam + StepLR — works well for small CNNs
# ─────────────────────────────────────────────

def train_small_model(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      epochs: int = 50,
                      lr: float = 1e-3,
                      device: str = "cpu") -> nn.Module:
    """
    Trains SmallCNN (MS) with Adam + StepLR.
    Target: ~72-75% on CIFAR-10.

    Args:
        model        : SmallCNN
        train_loader : augmented training DataLoader
        val_loader   : clean validation DataLoader
        epochs       : 50 epochs enough for SmallCNN to converge
        lr           : 1e-3 works well with Adam for small CNNs
        device       : 'cpu' or 'cuda'
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Step down lr at epoch 25 and 40
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 40], gamma=0.1)

    for epoch in range(epochs):
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
        val_acc = evaluate(model, val_loader, device)
        print(f"[MS] Epoch [{epoch+1:02d}/{epochs}]  "
              f"Loss: {train_loss/len(train_loader):.4f}  "
              f"Train: {train_correct/train_total:.4f}  "
              f"Val: {val_acc:.4f}")

    return model


# ─────────────────────────────────────────────
# STAGE 1b — Train ML (ResNet-18)
# SGD + CosineAnnealingLR — standard recipe
# Target: 93%+ on CIFAR-10
# ─────────────────────────────────────────────

def train_large_model(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      epochs: int = 200,
                      lr: float = 0.1,
                      device: str = "cpu") -> nn.Module:
    """
    Trains ResNet-18 (ML) with SGD + momentum + cosine schedule.
    This is the standard recipe that gets ResNet-18 to 93%+ on CIFAR-10.

    Args:
        model        : ResNet-18
        train_loader : augmented training DataLoader
        val_loader   : clean validation DataLoader
        epochs       : 100 epochs for full convergence
        lr           : 0.1 starting lr for SGD (standard for ResNets)
        device       : 'cpu' or 'cuda'
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    for epoch in range(epochs):
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

        # Print every 10 epochs to avoid too much output
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_acc = evaluate(model, val_loader, device)
            print(f"[ML] Epoch [{epoch+1:03d}/{epochs}]  "
                  f"Loss: {train_loss/len(train_loader):.4f}  "
                  f"Train: {train_correct/train_total:.4f}  "
                  f"Val: {val_acc:.4f}")

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
                         epochs: int = 30,
                         lr: float = 3e-4,
                         device: str = "cpu") -> nn.Module:
    """
    Fine-tune MS with GatekeeperLoss.
    MS should already be pre-trained (Stage 1) before calling this.
    ML is never touched here.

    Args:
        model_s      : pre-trained SmallCNN (MS)
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        alpha        : Gatekeeper trade-off in (0, 1)
                       paper sweeps: [0.1, 0.3, 0.5, 0.7, 0.9]
        num_classes  : number of output classes
        epochs       : 30 epochs — enough for confidence to reshape
        lr           : 3e-4 — slightly higher than before for faster reshaping
        device       : 'cpu' or 'cuda'
    """
    model_s = model_s.to(device)
    criterion = GatekeeperLoss(alpha=alpha, num_classes=num_classes)
    optimizer = optim.Adam(model_s.parameters(), lr=lr, weight_decay=1e-4)
    # Gentle decay in the second half of fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    print(f"\n── Gatekeeper Fine-tuning | alpha={alpha} ──")

    for epoch in range(epochs):
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

        scheduler.step()
        val_acc = evaluate(model_s, val_loader, device)
        print(f"  Epoch [{epoch+1:02d}/{epochs}]  "
              f"GK Loss: {total_loss/len(train_loader):.4f}  "
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
    print(f"Using device: {device}")

    # ── Transforms ──
    # Augmented for training — helps both MS and ML generalise better
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    # Clean for validation — no augmentation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=128,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(test_set,  batch_size=128,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Stage 1a: Train MS (SmallCNN) ──
    # Target: ~72-75% val accuracy
    print("\n=== Stage 1a: Training MS (SmallCNN) ===")
    model_s = SmallCNN(num_classes=10)
    model_s = train_small_model(model_s, train_loader, val_loader,
                                 epochs=50, lr=1e-3, device=device)
    torch.save(model_s.state_dict(), "model_s_pretrained.pth")
    print(f"Saved → model_s_pretrained.pth")

    # ── Stage 1b: Train ML (ResNet-18) ──
    # Target: ~93% val accuracy
    print("\n=== Stage 1b: Training ML (ResNet-18) ===")
    model_l = get_resnet18(num_classes=10)
    model_l = train_large_model(model_l, train_loader, val_loader,
                                 epochs=200, lr=0.1, device=device)
    torch.save(model_l.state_dict(), "model_l_pretrained.pth")
    print(f"Saved → model_l_pretrained.pth")

    # ── Stage 2: Fine-tune MS with Gatekeeper loss ──
    print("\n=== Stage 2: Gatekeeper Fine-tuning ===")
    for alpha in [0.9, 0.7, 0.5, 0.3, 0.1]:
        # Always reload fresh Stage 1 MS — each alpha starts from same baseline
        model_s_ft = SmallCNN(num_classes=10)
        model_s_ft.load_state_dict(
            torch.load("model_s_pretrained.pth", map_location=device))

        model_s_ft = finetune_gatekeeper(
            model_s_ft, train_loader, val_loader,
            alpha=alpha, num_classes=10,
            epochs=30, lr=3e-4, device=device
        )
        torch.save(model_s_ft.state_dict(), f"model_s_gk_alpha{alpha}.pth")
        print(f"Saved → model_s_gk_alpha{alpha}.pth")
