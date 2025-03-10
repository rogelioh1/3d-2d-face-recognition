Quick explantaion of the training process (written by claude):

## Model Training Process

The training process is primarily handled in `train.py` and relies on several key components:

### 1. Dataset Preparation

Before training begins, the system expects a specific data organization:
```
data/
  ├── train/
  │   ├── positive/  # Real 3D faces
  │   └── negative/  # Flat images/spoofing attempts
  ├── val/
  │   ├── positive/
  │   └── negative/
  └── test/
      ├── positive/
      └── negative/
```

The `DepthFaceDataset` class in `dataset.py` loads these images, which are depth maps (likely grayscale images where pixel intensity represents distance from the camera).

### 2. Data Transformation & Augmentation

In `train.py`, training data undergoes several transformations:
```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally (data augmentation)
    transforms.RandomRotation(10),  # Randomly rotate images up to 10 degrees (data augmentation)
    transforms.ToTensor()  # Convert images to PyTorch tensors
])
```

These transformations help improve model generalization by creating variations of the training data.

### 3. Model Architecture

The model defined in `model.py` is a standard CNN with:
- Input layer: Accepts single-channel 256×256 depth maps
- Three convolutional blocks, each with:
  - Conv2d layer (increasing channels: 1→32→64→128)
  - ReLU activation
  - MaxPool2d (downsampling)
- Flattening layer
- Two fully connected layers (128×32×32→256→num_classes)
- Dropout (0.5) for regularization

This architecture progressively extracts features from depth maps, from low-level details to higher-level features that distinguish 3D faces from flat images.

### 4. Training Loop

The training process in `train_model()` function follows this sequence:

1. **Setup**:
   ```python
   # Initialize model, loss function, and optimizer
   model = FaceShapeRecognitionModel(num_classes=len(LABEL_MAPPING))
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
   scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
   ```

2. **Training Loop**:
   ```python
   for epoch in range(NUM_EPOCHS):  # Default: 20 epochs
       model.train()  # Set model to training mode
       running_loss = 0.0
       
       for i, (images, labels) in enumerate(train_loader):
           # Move data to GPU if available
           images, labels = images.to(device), labels.to(device)
           
           # Forward pass
           optimizer.zero_grad()  # Reset gradients
           outputs = model(images)  # Get model predictions
           loss = criterion(outputs, labels)  # Calculate loss
           
           # Backward pass and optimization
           loss.backward()  # Compute gradients
           optimizer.step()  # Update weights
           running_loss += loss.item()
   ```

3. **Validation**:
   After each epoch, the model is evaluated on the validation set:
   ```python
   model.eval()  # Set model to evaluation mode
   correct = 0
   total = 0
   with torch.no_grad():  # Disable gradient calculation
       for images, labels in val_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)  # Get predicted class
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
   
   accuracy = 100 * correct / total
   ```

4. **Model Saving**:
   The best model (based on validation accuracy) is saved:
   ```python
   if accuracy > best_accuracy:
       best_accuracy = accuracy
       torch.save(model.state_dict(), MODEL_SAVE_PATH)
   ```

5. **Learning Rate Adjustment**:
   The learning rate is reduced every 10 epochs to fine-tune the model:
   ```python
   scheduler.step()  # Reduce learning rate according to schedule
   ```

### 5. Key Training Parameters

From `config.py`:
- `BATCH_SIZE = 16`: Number of images processed per iteration
- `NUM_EPOCHS = 20`: Number of complete passes through the training dataset
- `LEARNING_RATE = 0.001`: Initial learning rate for the optimizer
- Model outputs binary classification: "positive" (1) or "negative" (0)

## Training Flow Overview

1. Depth maps are loaded and preprocessed (resized, augmented, normalized)
2. The CNN processes batches of 16 images at a time
3. For each image, the model produces output logits
4. Cross-entropy loss compares predictions to ground truth labels
5. Gradients are calculated and model weights updated
6. This process repeats for 20 epochs
7. Every 10 epochs, the learning rate is reduced by 90%
8. The model with the best validation accuracy is saved

This training approach creates a model that can effectively distinguish between the depth patterns of real 3D faces and flat 2D images, providing the anti-spoofing capability critical to the system's security.
