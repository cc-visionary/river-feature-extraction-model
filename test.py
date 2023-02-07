import torchvision
import torchvision.transforms as transforms
import torch
import cv2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model=torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT') 
model.eval()
model.to(device)

image_path = './sample_images/9e9b4e4d952b8.jpg'

image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

image_tensor = transform(image)

print(image_tensor.shape)

cv2.imshow('original', image)
cv2.setWindowProperty('original', 1, cv2.WINDOW_NORMAL)
cv2.resizeWindow('original', 1000, 1000)
cv2.waitKey(0)
cv2.destroyAllWindows()

# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# pred = model(x)

# cv2.

# print(pred)