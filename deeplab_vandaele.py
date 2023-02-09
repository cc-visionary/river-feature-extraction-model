import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from deeplab.models import DeepLabV2_ResNet101_MSC
from utils import white_mask

def preprocessing(image, device):
    # Resize
    scale = 0.2
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(122.675),
            float(116.669),
            float(122.675),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image

classes = {0: 'non-river', 1: 'river'}

def load_model(device):
    model_path = './deeplab/weights/water_whole.pth'

    model = DeepLabV2_ResNet101_MSC(n_classes=len(classes))
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model

def load_image(image_path, device):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image, raw_image = preprocessing(image, device)

    return image, raw_image


def remove_non_river(image_path, device):
    model = load_model(device)
    image, raw_image = load_image(image_path, device)

    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(
        H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.detach().cpu().numpy()

    labelmap = np.argmax(probs, axis=0)
    # labels = np.unique(labelmap)

    labelmap = (~labelmap.astype(bool)).astype(int)
    w_mask = white_mask(labelmap)
    raw_image = cv2.addWeighted(raw_image, 1, w_mask, 1, 0)

    # # Show result for each class
    # rows = int(np.floor(np.sqrt(len(labels) + 1)))
    # cols = int(np.ceil((len(labels) + 1) / rows))

    # plt.figure(figsize=(10, 10))
    # ax = plt.subplot(rows, cols, 1)
    # ax.set_title("Input image")
    # ax.imshow(raw_image[:, :, ::-1])
    # ax.axis("off")

    # for i, label in enumerate(labels):
    #     mask = labelmap == label
    #     ax = plt.subplot(rows, cols, i + 2)
    #     ax.set_title(classes[label])
    #     ax.imshow(raw_image[..., ::-1])
    #     ax.imshow(mask.astype(np.float32), alpha=0.5)
    #     ax.axis("off")

    # plt.tight_layout()
    # plt.show()
    return raw_image