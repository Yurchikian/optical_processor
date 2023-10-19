import math
import torch
import torchvision
import numpy as np
import cv2
import imutils
import torchfields
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# device = 'cpu'

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
model = model.eval()

converter = torchvision.transforms.ToTensor()


def pad(size, multiple=8):
  # print("SIZE: ", size, ",", math.ceil(size / multiple), " -> ", math.ceil(size / multiple) * multiple)

  return math.ceil(size / multiple) * multiple


def pad_to_multiple_of(image, multiple=8):
  width, height, depth = image.shape
  
  desired_width = pad(width, multiple)
  desired_height = pad(height, multiple)

  image = cv2.copyMakeBorder(image, 0, desired_width - width, 0, desired_height - height, borderType=cv2.BORDER_REPLICATE)

  # print("new shape: ", image.shape)
  return image


def get_image(path):
  image = cv2.imread(path)
  # image = imutils.resize(image, width=512, height=512)
  # image = cv2.resize(image, (256, 256))

  image = imutils.resize(image, width=640, inter=cv2.INTER_LINEAR)
  image = pad_to_multiple_of(image, 8)

  return np.asarray(image)

def calc(a, b):
  a = converter(a).to(device)
  b = converter(b).to(device)

  return model(b[None], a[None])[-1]

def apply_flow(image, flow):
  image_tensor = converter(image).to(device)
  displacement_field = flow.field().from_pixels()
  # print("displacement_field. From ", displacement_field.min().item(), " to ", displacement_field.max().item())

  displaced_image = (displacement_field)(image_tensor).detach().numpy()
  displaced_image = np.moveaxis(displaced_image, 0, 2)

  return displaced_image

start_idx = 281
finish_idx = 297

def name_idx(idx):
  return 'in/video_%05d.jpg' % idx

image_root = get_image(name_idx(start_idx))

cv2.imwrite('A00.jpg', image_root)


# print("Processing image", )

# image_b = get_image(name_idx(51))

# flow = calc(image_root, image_b)

# image_displaced = apply_flow(image_root, flow)

# cv2.imwrite('A01_%05d.jpg' % 1, image_displaced * 255)



for i in range(finish_idx - start_idx):
  idx = start_idx + i + 1
  image_name = name_idx(idx)

  print("Processing image", image_name)

  image_b = get_image(image_name)
  flow = calc(image_root, image_b)[-1]
  image_displaced = apply_flow(image_root, flow)
  cv2.imwrite('A01_%05d.jpg' % i, image_displaced * 255)

# plt.figure()
# plt.imshow(displaced_image)
# plt.figure()
# plt.imshow(image_b)
# plt.show()
