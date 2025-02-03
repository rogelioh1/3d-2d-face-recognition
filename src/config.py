DATASET_PATH = "data"
TRAIN_POSITIVE_PATH = f"{DATASET_PATH}/train/positive"
TRAIN_NEGATIVE_PATH = f"{DATASET_PATH}/train/negative"
VAL_POSITIVE_PATH = f"{DATASET_PATH}/val/positive"
VAL_NEGATIVE_PATH = f"{DATASET_PATH}/val/negative"
MODEL_SAVE_PATH = "facedepthmodel.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEPTH_WEIGHT = 0.5

LABEL_MAPPING = {
    "positive": 1,
    "negative": 0
}