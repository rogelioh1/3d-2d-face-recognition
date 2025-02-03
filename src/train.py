import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DepthFaceDataset
from model import FaceShapeRecognitionModel
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, DATASET_PATH, LABEL_MAPPING

from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

def train_model():
    # Define the transform with data augmentation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    # Initialize dataset and dataloaders
    train_dataset = DepthFaceDataset(os.path.join(DATASET_PATH, 'train'), transform=transform)
    val_dataset = DepthFaceDataset(os.path.join(DATASET_PATH, 'val'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = FaceShapeRecognitionModel(num_classes=len(LABEL_MAPPING))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Check if GPU is available and move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {running_loss/len(train_loader):.4f}')

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'Best model saved to {MODEL_SAVE_PATH}')

        # Step the scheduler
        scheduler.step()

    print(f'Best Validation Accuracy: {best_accuracy:.2f}%')

if __name__ == "__main__":
    train_model()