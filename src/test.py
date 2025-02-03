import os
import torch
from torch.utils.data import DataLoader
from dataset import DepthFaceDataset
from model import FaceShapeRecognitionModel
from config import BATCH_SIZE, MODEL_SAVE_PATH, DATASET_PATH, LABEL_MAPPING
from torchvision import transforms

def test_model():
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    # Initialize dataset and dataloader
    test_dataset = DepthFaceDataset(os.path.join(DATASET_PATH, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model and load trained weights
    model = FaceShapeRecognitionModel(num_classes=len(LABEL_MAPPING))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # Check if GPU is available and move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    test_model()
