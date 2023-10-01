import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    """
    Gets images and labels from labels.txt, and loads images from data_dir
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = {}

        filepath = os.path.join(data_dir, "labels.txt")

        if not os.path.exists(filepath):
            print(f"labels.txt not found in {data_dir}")
        else:
            with open(filepath, "r") as f:
                for line in f:
                    image_name = line.strip().split(",")[0]
                    self.labels[image_name] = [list(map(float, line.strip().split(",")[1:]))]
            self.images = list(self.labels.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.data_dir, img_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_matrix = torch.tensor(self.labels[img_name]) 
        return image, label_matrix

def get_transforms(augment=True):
    basic_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if augment:
        augmented_transforms = [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ]
        return transforms.Compose(augmented_transforms + basic_transforms)
    else:
        return transforms.Compose(basic_transforms)