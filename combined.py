from deeplab.deeplab import predict_single

from mask_rcnn.mrcnn import utils
import mask_rcnn.mrcnn.model as modellib

if __name__ == "__main__":
    # config_path = './deeplab/weights/water.yaml'
    # model_path = './deeplab/weights/laura_whole.pth'
    # image_path = './sample_images/9e9b4e4d952b8.jpg'
    # predict_single(config_path, model_path, image_path)
    pass
    
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