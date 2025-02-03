import os
import cv2
import numpy as np
import torch

def load_depth_map(image_path):
    depth_map = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise FileNotFoundError(f"Depth map not found at {image_path}")
    return depth_map

def preprocess_depth_map(depth_map, target_size=(256, 256)):
    depth_map = cv2.resize(depth_map, target_size)
    depth_map = depth_map.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return depth_map

def to_tensor(depth_map):
    return torch.tensor(depth_map).unsqueeze(0)  # Add channel dimension

def visualize_depth_map(depth_map):
    depth_map = (depth_map * 255).astype(np.uint8)
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()