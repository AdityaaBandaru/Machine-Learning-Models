import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import config
from utils import get_dataloaders
from model import ResNet20
from ablation_model import ResNet20NoSkip

# CIFAR-10 Class Names
CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def get_predictions(model, data_loader, device):
    """Evaluates model and collects all predictions and true targets."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    return np.array(all_preds), np.array(all_targets)


def main():
    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running model comparison on device: {device}\n")

    # 2. Load Data
    _, test_loader = get_dataloaders(
        config["data_dir"], config["batch_size"], config["num_workers"]
    )

    # 3. Load ResNet-20 (With Skip Connections)
    resnet = ResNet20().to(device)
    resnet.load_state_dict(torch.load("resnet20_best.pth", map_location=device))
    
    # 4. Load PlainNet-20 / ResNet20NoSkip (Without Skip Connections)
    # Note: If your saved file is named 'baseline_best.pth', update the filename below
    plainnet = ResNet20NoSkip().to(device)
    plainnet.load_state_dict(torch.load("resnet20_noskip_best.pth", map_location=device))

    # 5. Generate Predictions
    print("Evaluating ResNet-20...")
    resnet_preds, targets = get_predictions(resnet, test_loader, device)
    
    print("Evaluating PlainNet-20 (No Skip)...")
    plain_preds, _ = get_predictions(plainnet, test_loader, device)

    # 6. Calculate Accuracies
    resnet_acc = (resnet_preds == targets).mean() * 100
    plain_acc = (plain_preds == targets).mean() * 100

    print("\n" + "=" * 45)
    print(f"{'Model':<25} | {'Test Accuracy':<15}")
    print("=" * 45)
    print(f"{'ResNet-20 (With Skip)':<25} | {resnet_acc:.2f}%")
    print(f"{'PlainNet-20 (No Skip)':<25} | {plain_acc:.2f}%")
    print(f"{'Accuracy Gap':<25} | +{resnet_acc - plain_acc:.2f}%")
    print("=" * 45 + "\n")

    # 7. Plot Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cm_resnet = confusion_matrix(targets, resnet_preds)
    cm_plain = confusion_matrix(targets, plain_preds)

    sns.heatmap(cm_resnet, ax=axes[0], annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    axes[0].set_title(f"ResNet-20 (Skip Connections)\nAccuracy: {resnet_acc:.2f}%", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True Label")

    sns.heatmap(cm_plain, ax=axes[1], annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    axes[1].set_title(f"PlainNet-20 (No Skip Connections)\nAccuracy: {plain_acc:.2f}%", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig("model_comparison_matrix.png", dpi=300)
    print("Saved comparison confusion matrices to 'model_comparison_matrix.png'!")
    plt.show()


if __name__ == "__main__":
    main()
    