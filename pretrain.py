import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.pretrain.myclip import MyCLIP
from src.pretrain.symmetric_loss import SymmetricLoss
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="pretrain")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-pretraining")

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
    #       CLIP
    # ------------------
    myclip = MyCLIP(meg_dropout=args.meg_dropout).to(device)
    myclip = torch.nn.DataParallel(myclip, device_ids=list(range(args.num_gpus)))

    # ------------------
    # Optimizer & Scheduler
    # ------------------
    optimizer = torch.optim.Adam(
        myclip.module.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # ------------------
    #   Start training
    # ------------------
    min_val_loss = 0
    criterion = SymmetricLoss().to(device)

    torch.backends.cudnn.benchmark = True

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, val_loss = [], []

        myclip.train()

        for meg, y, subject_idxs, image in tqdm(train_loader, desc="Train"):
            meg, y, image = meg.to(device), y.to(device), image.to(device)

            optimizer.zero_grad()

            logits_per_image, logits_per_meg = myclip(image, meg, subject_idxs)
            loss = criterion(logits_per_image, logits_per_meg, y)
            l2 = torch.tensor(0.0, requires_grad=True)
            for w in myclip.parameters():
                l2 = l2 + torch.norm(w) ** 2
            loss = loss + args.alpha * l2
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()

        myclip.eval()
        for meg, y, subject_idxs, image in tqdm(val_loader, desc="Validation"):
            meg, y, image = meg.to(device), y.to(device), image.to(device)

            with torch.no_grad():
                logits_per_image, logits_per_meg = myclip(image, meg, subject_idxs)

            loss = criterion(logits_per_image, logits_per_meg, y)
            l2 = torch.tensor(0.0, requires_grad=True)
            for w in myclip.parameters():
                l2 = l2 + torch.norm(w) ** 2
            loss = loss + args.alpha * l2
            val_loss.append(loss.item())

        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | val loss: {np.mean(val_loss):.3f}"
        )
        torch.save(myclip.module.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "val_loss": np.mean(val_loss),
                }
            )

        if np.mean(val_loss) < min_val_loss:
            cprint("New best.", "cyan")
            torch.save(
                myclip.module.state_dict(), os.path.join(logdir, "model_best.pt")
            )
            min_val_loss = np.mean(val_loss)


if __name__ == "__main__":
    run()
