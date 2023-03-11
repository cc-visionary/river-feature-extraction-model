import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from deeplab.models import DeepLabV2_ResNet101_MSC
from utils import white_mask
from scipy import ndimage

classes = {0: 'non-river', 1: 'river'}

def load_model(device):
    model_path = 'weights/water_whole.pth'

    model = DeepLabV2_ResNet101_MSC(n_classes=len(classes))
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model)
    model.eval()
    model.to(device)

    return model

"""
returns output_image, mask
"""
def remove_non_river(image, raw_image, device):
    model = load_model(device)

    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(
        H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.detach().cpu().numpy()

    labelmap = np.argmax(probs, axis=0)

    labelmap = (~labelmap.astype(bool)).astype(int)
    labelmap = ndimage.binary_fill_holes(1 - labelmap).astype(int)
    labelmap = 1 - ndimage.binary_fill_holes(labelmap).astype(int)
    labelmap = ndimage.binary_dilation(labelmap).astype(int)
    w_mask = white_mask(labelmap)
    raw_image = cv2.addWeighted(raw_image, 1, w_mask, 1, 0)

    return raw_image, labelmap