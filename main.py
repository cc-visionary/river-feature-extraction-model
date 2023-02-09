import cv2

from deeplab_vandaele import remove_non_river
from mask_rcnn_coco import remove_boats
from utils import get_device

if __name__ == "__main__":
    image_path = './sample_images/868df0899e668.jpg'
    device = get_device()
    image = remove_non_river(image_path, device)
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