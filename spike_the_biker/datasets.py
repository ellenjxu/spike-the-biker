import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from torchvision.io import read_image


def rotate_trajectory(traj, yaw):
    """Rotate a trajectory by yaw."""
    rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    rotated_traj = np.dot(rotation_matrix, traj.T).T
    return rotated_traj


class TrajectoryDataset(Dataset):
    def __init__(self, root_dir, N, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "processed_images"
        self.N = N
        self.trajectory_data = np.loadtxt(self.root_dir / "trajectories.txt")
        self.valid_indices = self.get_valid_indices()
        self.transform = transform

    def __len__(self):
        return len(self.valid_indices)

    def get_valid_indices(self):
        valid_indices = []

        start, end = 0, 0
        for i, end_of_sequence in enumerate(self.trajectory_data[:, -1]):
            if end_of_sequence:
                end = i
                if end - start >= self.N:  # if we have at least one valid subsequence
                    # Add the start of each valid subsequence of N steps
                    for subseq_start in range(
                        start, end - self.N + 1
                    ):  # +1 bc range end is exclusive
                        valid_indices.append(subseq_start)
                start = end + 1  # set the start of the next sequence

        return valid_indices

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        # Load image
        img_idx = real_idx + 1  # shifted by one
        img_pth = self.image_dir / f"{str(img_idx).zfill(4)}.jpg"
        image = read_image(str(img_pth))

        # collate on the fly so don't need
        # if self.transform:
        #     image = self.transform(image)

        # Load and process trajectory
        trajectory_data = self.trajectory_data[
            real_idx : real_idx + self.N
        ]  # fetch N future steps

        # Extract car positions (x, y) and yaw
        car_positions = trajectory_data[:, :2]
        yaw = trajectory_data[0, 2]  # fetch yaw at the current step

        # Rotate the future car positions into the car's frame
        relative_positions = car_positions - car_positions[0]  # make relative to current position
        rotated_positions = rotate_trajectory(relative_positions, yaw)

        # get the steering
        steering = trajectory_data[:, 3].reshape(-1, 1)
        # throttle = trajectory_data[:, 5].reshape(-1, 1)
        trajectory = np.column_stack((rotated_positions, steering))

        # TODO return steering and throttle as different keys
        # do the hstack in the neural net instead?
        # allows using this same code for steering only or steering + throttle
        return (image, trajectory.flatten().astype(np.float32))


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
        # transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
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