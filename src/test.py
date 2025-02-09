import os
import torch
from torch.utils.data import DataLoader
from dataset import DepthFaceDataset
from model import FaceShapeRecognitionModel
from config import BATCH_SIZE, DATASET_PATH, LABEL_MAPPING
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

# Replace MODEL_SAVE_PATH definition
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.pth')

def test_model():
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    # Initialize dataset and dataloader
    test_dataset = DepthFaceDataset(os.path.join(DATASET_PATH, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model and load trained weights from the root project directory
    model = FaceShapeRecognitionModel(num_classes=len(LABEL_MAPPING))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # Check if GPU is available and move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # New: Compute and display detailed classification metrics
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    test_model()