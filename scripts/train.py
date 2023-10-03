import hydra
import torch
import torch.onnx
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate, random_split
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from tqdm import tqdm

import wandb
from spike_the_biker.datasets import TrajectoryDataset, get_transforms
from spike_the_biker.models import EfficientNet
from spike_the_biker.viz import make_vid_from_2d_traj

import numpy as np

def get_outputs(model, dl, criterion=None):
    preds = []
    labels = []
    losses = []

    model = model.eval()

    with torch.no_grad():
        model = model.eval()
        for images, label in dl:
            images, label = images.to("cuda"), label.to("cuda")
            pred = model(images)

            if criterion is not None:
                loss = criterion(pred, label)
                losses.append(loss.cpu())

            preds.append(pred.cpu())
            labels.append(label.cpu())

    model = model.train()

    return preds, labels, losses

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    
    # set up wandb
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project, save_code=True, job_type=cfg.wandb.job_type, config=config
    )
    
    #  split
    ds200k = TrajectoryDataset(root_dir=cfg.dataset.data_dir, N=cfg.dataset.N, transform=None)
    ds150k = TrajectoryDataset(root_dir=cfg.dataset.data_dir_2, N=cfg.dataset.N, transform=None)
    ds = torch.utils.data.ConcatDataset([ds200k, ds150k])

    viz_ds, dataset = torch.utils.data.Subset(ds, range(0, 300)), torch.utils.data.Subset(
        ds, range(300, len(ds))
    )
    
    val_len = int(cfg.dataset.validation_split * len(dataset))
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # then apply augmentations
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    def collate_transforms(batch, transform=None):  # apply transforms on the fly using collate
      images, labels = zip(*batch)
      if transform:
        images = [transform(i) for i in images]
      return default_collate(list(zip(images, labels))) # back into type Tensor
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, 
                              collate_fn=lambda x: collate_transforms(x, train_transform),
                              num_workers=6,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, 
                            collate_fn=lambda x: collate_transforms(x, val_transform),
                            num_workers=6,
                            pin_memory=True)
    viz_dl = DataLoader(viz_ds, batch_size=cfg.dataset.batch_size, shuffle=False, 
                            collate_fn=lambda x: collate_transforms(x, train_dataset), pin_memory=True)
    
    # efficientnet
    model = EfficientNet(out_size=cfg.model.out_size).to("cuda")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=100, verbose=True, factor=0.5, min_lr=1e-6)

    # Training loop
    for epoch in range(cfg.model.epochs):
        print(f"epoch: {epoch}")
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            scheduler.step(loss)
            total_loss += loss.item()
            
            wandb.log({"train/loss": loss, "train/lr": optimizer.param_groups[0]['lr']})
        
        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        _, _, losses = get_outputs(model, val_loader, criterion)
        avg_val_loss = torch.stack(losses).mean()

        # # viz
        # preds, labels, _ = get_outputs(model, viz_dl, None) # B, OUT_LEN
        # preds = torch.cat(preds, axis=0).view(-1, cfg.dataset.N, 3) # x, y, steer
        # labels = torch.cat(labels, axis=0).view(-1, cfg.dataset.N, 3)

        # pred_traj = preds[:, :2]
        # gt_traj = labels[:, :2]
        # vid_2d = make_vid_from_2d_traj(pred_traj, gt_traj)
        # wandb.log(
        #     {
        #         f"train/2d": wandb.Video(vid_2d.transpose(0, 3, 1, 2), fps=20),
        #     }
        # )
                
        print(f"\nEpoch {epoch+1}/{cfg.model.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        wandb.log({"Train Loss": avg_train_loss, "Validation Loss": avg_val_loss})

    # Save model
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model.cpu(), dummy_input, "models/efficientnet_b0_200k_xy_theta_150k.onnx")

    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("models/efficientnet_b0_200k_xy_theta_150k.onnx")
    run.log_artifact(artifact)

    run.finish()



if __name__ == "__main__":
    main()
