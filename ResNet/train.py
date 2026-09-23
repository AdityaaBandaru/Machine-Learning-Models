import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from model import ResNet20
from utils import evaluate, get_dataloaders, plot_metrics, train_one_epoch


def main():
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        config["data_dir"], config["batch_size"], config["num_workers"]
    )

    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[config["epochs"] // 2, 3 * config["epochs"] // 4],
        gamma=0.1,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "resnet20_best.pth")

        print(
            f"Epoch {epoch:03d}/{config['epochs']} | "
            f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc * 100:.2f}% | "
            f"best {best_acc * 100:.2f}% | "
            f"lr {scheduler.get_last_lr()[0]:.4f}"
        )

    plot_metrics(history, path="training_curves.png")
    print(f"Best test accuracy: {best_acc * 100:.2f}%")
    print("Saved plots to training_curves.png")



if __name__ == "__main__":
    main()
