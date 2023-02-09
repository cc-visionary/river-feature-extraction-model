import torch
import numpy as np

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print("Device:", torch.cuda.get_device_name(
            torch.cuda.current_device()))
    else:
        device = "cpu"
        print("Device: CPU")
        
    return device

def white_mask(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = [255, 255, 255]
    mask = np.stack([r, g, b], axis=2)

    return mask