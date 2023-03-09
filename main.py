import cv2
import numpy as np
import torch
import warnings

from deeplab_vandaele import remove_non_river, load_model
from mask_rcnn_coco import remove_boats
from utils import get_device, load_image, preprocessing
from dataset import RiverDataset
from torch.utils.data import random_split, DataLoader

if __name__ == "__main__":
    image_path = 'images/images/868df0899e668.jpg'
    device = get_device()
    img, raw_img = load_image(image_path, device)
    image = remove_non_river(img, raw_img, device)
    image = remove_boats(image, device)

    cv2.imshow('predicted', image)
    cv2.setWindowProperty('predicted', 1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow('predicted', 1000, 1000)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
TODO:
- Modify to set river mask to visible and remove non-river
- Validate on whole dataset (IoU)

- Use image masked from river segmentation model to mask out boats
- Validate segmentation (IoU)

- Experiment on Image Processing Techniques
- Image processing with results
- Validate image processing
"""
