import torchvision
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
import random

from PIL import Image
from torchvision.utils import draw_bounding_boxes

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    weights='DEFAULT')
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT.meta[
    'categories']

image_path = './sample_images/868df0899e668.jpg'

image = Image.open(image_path)

def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [
        255, 0, 255], [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image ==
                                    1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

# Define a transform to convert PIL
# image to a Torch tensor
image = F.pil_to_tensor(image)
image_tensor = F.convert_image_dtype(image)

prediction = model([image_tensor])
pred_score = list(prediction[0]['scores'].detach().numpy())
threshold = 0.5
pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

masks = (prediction[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i]
              for i in list(prediction[0]['labels'].numpy())]
pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
              for i in list(prediction[0]['boxes'].detach().numpy())]

masks = masks[:pred_t+1]
pred_boxes = pred_boxes[:pred_t+1]
pred_class = pred_class[:pred_t+1]

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for i in range(len(masks)):
    rgb_mask = random_colour_masks(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    cv2.rectangle(img, [int(val) for val in pred_boxes[i][0]], [int(val) for val in pred_boxes[i][1]],color=(0, 255, 0), thickness=3)
    cv2.putText(img,pred_class[i], [int(val) for val in pred_boxes[i][0]], cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),thickness=3)

cv2.imshow('predicted', img)
cv2.setWindowProperty('predicted', 1, cv2.WINDOW_NORMAL)
cv2.resizeWindow('predicted', 1000, 1000)
cv2.waitKey(0)
cv2.destroyAllWindows()
