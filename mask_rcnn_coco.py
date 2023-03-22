import torchvision
import torch                                                        
import cv2
import numpy as np
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights 
import random

from PIL import Image

from utils import white_mask

COCO_INSTANCE_CATEGORY_NAMES = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.meta['categories']

def load_model(device):
    model_path = 'weights/maskrcnn_resnet50_fpn_coco.pth'   
    weights = torch.load(model_path)

    model = maskrcnn_resnet50_fpn(pretrained_backbone=False)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    return model

def load_image(image, device):
    # Define a transform to convert PIL
    # image to a Torch tensor
    image = F.to_tensor(image)
    image_tensor = F.convert_image_dtype(image)
    image_tensor = image_tensor.to(device)

    return image_tensor

"""
returns output_image, mask
"""
def remove_boats(image, device):
    model = load_model(device)
    image_tensor = load_image(image, device)

    prediction = model([image_tensor])
    pred_score = list(prediction[0]['scores'].detach().cpu().numpy())
    threshold = 0.5
    boat_mask = np.zeros((360, 640))

    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

        masks = (prediction[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(prediction[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                    for i in list(prediction[0]['boxes'].detach().cpu().numpy())]

        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

        for i in range(len(masks)):
            if(pred_class[i] == 'boat'):
                w_mask = white_mask(masks[i])
                image = cv2.addWeighted(image, 1, w_mask, 1, 0)
                boat_mask = np.logical_or(boat_mask, masks[i])
    except:
        pass
    
    return image, boat_mask
