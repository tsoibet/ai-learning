import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

img = read_image('./data/cat.jpg')
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img.permute(1,2,0))

# Q1
print("Information of the image (RGB channel, Height, Width):", img.shape)
print("First pixel:", img[:, 0, 0])
new_img = img[:, :, 80:560]
plt.subplot(1, 3, 2)
plt.title("Q1")
plt.imshow(to_pil_image(new_img))

# Q2
new_img = 255 - img
plt.subplot(1, 3, 3)
plt.title("Q2")
plt.imshow(to_pil_image(new_img))

plt.show()