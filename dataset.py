import os
import numpy as np
import cv2

from torch.utils.data import Dataset
from PIL import Image
from scipy import ndimage

class RiverDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.images_path = "images"
        self.masks_path = "masks"

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(
            sorted(os.listdir(os.path.join(root, self.images_path))))
        self.masks = list(
            sorted(os.listdir(os.path.join(root, self.masks_path))))

    def __getitem__(self, index):
        # load images ad masks
        img_path = os.path.join(self.root, self.images_path, self.imgs[index])
        mask_path = os.path.join(self.root, self.masks_path, self.masks[index])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Subtract mean values
        img = img.astype(np.float32)
        img -= np.array(
            [
                float(122.675),
                float(116.669),
                float(104.008),
            ]
        )
        img =  Image.fromarray(img, mode='RGB').resize((640, 360), Image.ANTIALIAS)

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        mask = mask.resize((640, 360), Image.ANTIALIAS)

        mask = np.array(mask)

        # retain only the area of the river
        obj_ids = np.array([[102, 255, 102]])

        # split the color-encoded mask into a set
        # of binary masks
        mask = np.any(mask == obj_ids[:, None, None], axis=(3))
        mask = np.transpose(mask, (1, 2, 0))[:, :, 0]
        mask = ndimage.binary_fill_holes(mask).astype(int)
        

        return self.imgs[index].split('.')[0], img.astype(np.float32), mask.astype(np.int8)

    def __len__(self):
        return len(self.imgs)
