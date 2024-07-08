import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="prepretrain")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-prepretraining")

    device = torch.device("cuda")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }

    print("Loading datasets...")

    train_set = ThingsMEGDataset("train", args.data_dir, with_image=True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir, with_image=True)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)

    print("Datasets loaded.")

    # ------------------
    #       ResNet34
    # ------------------
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    for i, param in enumerate(model.parameters()):
        if i < args.num_freeze_layers:
            param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, train_set.num_classes)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    # ------------------
    # Optimizer & Scheduler
    # ------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(device)

    torch.backends.cudnn.benchmark = True

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()

        for _, y, _, image in tqdm(train_loader, desc="Train"):
            image, y = image.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(image)
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())
        scheduler.step()

        model.eval()
        for _, y, _, image in tqdm(val_loader, desc="Validation"):
            image, y = image.to(device), y.to(device)

            with torch.no_grad():
                y_pred = model(image)

            loss = F.cross_entropy(y_pred, y)
            val_loss.append(loss.item())
            val_acc.append(accuracy(y_pred, y).item())

        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        )
        torch.save(model.module.state_dict(), os.path.join(logdir, "resnet34_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "train_acc": np.mean(train_acc),
                    "val_loss": np.mean(val_loss),
                    "val_acc": np.mean(val_acc),
                }
            )

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(
                model.module.state_dict(), os.path.join(logdir, "resnet34_best.pt")
            )
            max_val_acc = np.mean(val_acc)


if __name__ == "__main__":
    run()
