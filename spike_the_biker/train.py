import hydra
import torch
import torch.onnx
from torch.utils.data import DataLoader, random_split, default_collate
import torchvision.transforms as transforms
import wandb
from omegaconf import OmegaConf
from models import EfficientNet
from datasets import CustomDataset, get_transforms
from tqdm import tqdm

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    
    # set up wandb
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project, save_code=True, job_type=cfg.wandb.job_type, config=config
    )

    #  split
    dataset = CustomDataset(data_dir=cfg.dataset.data_dir, transform=None)
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
                              collate_fn=lambda x: collate_transforms(x, train_transform))
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, 
                            collate_fn=lambda x: collate_transforms(x, val_transform))

    # efficientnet
    model = EfficientNet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    # Training loop
    for epoch in tqdm(range(cfg.model.epochs)):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
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
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}/{cfg.model.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        wandb.log({"Train Loss": avg_train_loss, "Validation Loss": avg_val_loss})

    # Save model
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "models/efficientnet.onnx")
    wandb.save("efficientnet.onnx")

    wandb.finish()

if __name__ == "__main__":
    main()