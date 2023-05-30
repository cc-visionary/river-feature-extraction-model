import torch
import numpy as np
import cv2

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


def preprocessing(image, device):
    # Resize
    image = cv2.resize(image, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(104.008),
            float(116.669),
            float(122.675),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def load_image(image_path, device):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, raw_image = preprocessing(image, device)

    return image, raw_image

def compute_iou(groundtruth_box, detection_box):
    g_xmin, g_ymin, g_xmax, g_ymax = groundtruth_box
    d_xmin, d_ymin, d_xmax, d_ymax = detection_box
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def get_precision_recall_f1(groundtruth_boxes, predicted_boxes, iou_value):
    """
    ground= array of ground-truth contours.
    preds = array of predicted contours.
    iou_value= iou treshold for TP and otherwise.

    https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py
    """
    matches = []
    for i in range(len(groundtruth_boxes)):
        for j in range(len(predicted_boxes)):
            iou = compute_iou(groundtruth_boxes[i], predicted_boxes[j])

            if(iou >= iou_value):
                matches.append([i, j, iou])
    matches = np.array(matches)
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(groundtruth_boxes)):
        if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
            tp += 1
        else:
            fn += 1
            
    for i in range(len(predicted_boxes)):
        if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
            fp += 1
    try:
        # lets add this section just to print the results
        print("TP:", tp, "\t FP:", fp, "\t FN:", fn, "\t GT:", len(groundtruth_boxes))
        precision = round(tp/(tp+fp), 3)
        recall = round(tp/(tp+fn), 3)
        f1 = round(2*((precision*recall)/(precision+recall)), 3)
        print("Precision:", precision, "\t Recall:", recall, "\t F1 score:", f1)
    except ZeroDivisionError:
        if(len(groundtruth_boxes) == fn and fn != 0):
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = -1
            recall = -1
            f1 = -1
        print("Precision:", precision, "\t Recall:", recall, "\t F1 score:", f1)

    return tp, fp, fn, precision, recall, f1
