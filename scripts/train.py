import hydra
import torch
import torch.onnx
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate, random_split
from tqdm import tqdm

import wandb
from spike_the_biker.datasets import TrajectoryDataset, get_transforms
from spike_the_biker.models import EfficientNet
from spike_the_biker.viz import make_vid_from_2d_traj, plot_traj

import numpy as np

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    
    # set up wandb
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project, save_code=True, job_type=cfg.wandb.job_type, config=config
    )
    
    #  split
    dataset = TrajectoryDataset(root_dir=cfg.dataset.data_dir, N=cfg.dataset.N, transform=None)

    viz_ds, ds = torch.utils.data.Subset(dataset, range(0, 300)), torch.utils.data.Subset(
        dataset, range(300, len(dataset))
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
                              collate_fn=lambda x: collate_transforms(x, train_transform), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, 
                            collate_fn=lambda x: collate_transforms(x, val_transform), pin_memory=True)
    viz_dl = DataLoader(
        viz_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # efficientnet
    model = EfficientNet(out_size=cfg.model.out_size).to("cuda")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    viz(viz_dl, model)

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
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            model = model.eval()
            for images, labels in val_loader:
                images, labels = images.to("cuda"), labels.to("cuda")
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                viz(viz_dl, model)
                viz(val_loader, model, key="test")
            model = model.train()
                
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}/{cfg.model.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        wandb.log({"Train Loss": avg_train_loss, "Validation Loss": avg_val_loss})

    # Save model
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "models/efficientnet.onnx")
    wandb.save("efficientnet.onnx")

    wandb.finish()

def viz(dl, model, key="val"):
    model.eval()
    images = []
    gt_traj = []
    pred_traj = []
    with torch.no_grad():
        for idx, batch in enumerate(dl):
            # forward
            x = batch["image"]

            if key != "test":
                y = batch["trajectory"]
                gt_traj.append(y.cpu().numpy())

                speed = batch["speed"]
            else:
                # dummy speed for test ds bc we didn't collect speed
                # TODO would be nice to record the real speed
                # speed is between -1, 1 so lets just use zero
                speed = torch.zeros((x.size(0), 1))
                speed = speed.to(x.device)

            y_hat = model(x, speed)

            # TODO this is ugly can we fix it?
            pred_traj.append(y_hat.cpu().numpy())
            images.append(x.cpu().numpy())

    model.train()

    # make the viz
    images = np.concatenate(images, axis=0)
    images = np.ascontiguousarray(images.transpose(0, 2, 3, 1))  # cv2 images are WHC

    # denormalize images
    images_min, images_max = images.min(), images.max()
    images = ((images - images_min) / (images_max - images_min) * 255).astype(np.uint8)

    if key != "test":
        gt_traj = np.concatenate(gt_traj, axis=0).reshape(
            images.shape[0], -1, 4
        )  # (images, nb points traj, xy)
    pred_traj = np.concatenate(pred_traj, axis=0).reshape(images.shape[0], -1, 4)

    for idx in range(images.shape[0]):
        if key != "test":  # only plot gt when we have it
            plot_traj(images[idx], gt_traj[idx, :, 0:2], color=(0, 255, 0))
        plot_traj(images[idx], pred_traj[idx, :, 0:2], color=(0, 0, 255))

    if key != "test":  # plot gt and pred
        vid_2d = make_vid_from_2d_traj(pred_traj, gt_traj)
    else:
        vid_2d = make_vid_from_2d_traj(pred_traj)

    wandb.log(
        {
            f"{key}/viz": wandb.Video(
                images.transpose(0, 3, 1, 2), fps=10
            ),  # actual fps is 40 but we want to viz more slowly TODO should be configured somewhere
            f"{key}/2d": wandb.Video(vid_2d.transpose(0, 3, 1, 2), fps=10),
        }
    )


if __name__ == "__main__":
    main()
