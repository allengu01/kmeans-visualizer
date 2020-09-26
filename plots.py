import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

img = cv2.imread('images/minimalist_landscape.jpg')
resized_img = cv2.resize(img, (20, 20))

fig = plt.figure(figsize = (8, 6))
ax = plt.axes(projection='3d')

def split_rgb(image):
    b, g, r = cv2.split(image)
    return r, g, b

def flatten(arr):
    n = arr.size
    return arr.reshape([n,])

img_r, img_g, img_b = split_rgb(resized_img)
flatten_img_r, flatten_img_g, flatten_img_b = list(map(flatten, [img_r, img_g, img_b]))
pixels = np.stack([flatten_img_r, flatten_img_g, flatten_img_b], axis=1)
print(pixels.shape)

for r, g, b in pixels:
    ax.scatter3D(r, g, b, color = (r / 255, g / 255, b / 255))

plt.show()

