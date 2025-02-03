import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import LABEL_MAPPING

class DepthFaceDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = []
        self.labels = []
        self._prepare_data()

    def _prepare_data(self):
        for label in ["positive", "negative"]:
            label_path = os.path.join(self.dataset_path, label)
            if os.path.exists(label_path):
                for person_folder in os.listdir(label_path):
                    person_path = os.path.join(label_path, person_folder)
                    if os.path.isdir(person_path):
                        self._load_depth_maps(person_path, LABEL_MAPPING[label])

    def _load_depth_maps(self, person_path, label):
        for root, _, files in os.walk(person_path):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):  # Assuming depth maps are in these formats
                    img_path = os.path.join(root, file)
                    self.data.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]  # Fix attribute name
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image at {image_path}: {e}")
            image = None

        if image is None:
            raise ValueError(f"Image at index {idx} from path {image_path} could not be loaded.")

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Example of transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming single channel depth maps
])