import os
import imageio

xray_image = imageio.v3.imread("numpy/00000011_000.png")

import numpy as np

from scipy import ndimage

import time

print("==================")
print("Experiment: x-ray-image.py\n")

time_list = []
for _ in range(50):
    t0 = time.time()
    xray_image_laplace_gaussian = ndimage.gaussian_laplace(xray_image, sigma=1)
    time_list.append(time.time()-t0)
print(f"Laplacian-Gaussian takes {sum(time_list)/len(time_list)}s on average")

time_list = []
for _ in range(50):
    t0 = time.time()
    x_ray_image_gaussian_gradient = ndimage.gaussian_gradient_magnitude(xray_image, sigma=2)
    time_list.append(time.time()-t0)
print(f"Gaussian gradient magnitude takes {sum(time_list)/len(time_list)}s on average")


time_list = []
for _ in range(50):
    t0 = time.time()
    x_sobel = ndimage.sobel(xray_image, axis=0)
    y_sobel = ndimage.sobel(xray_image, axis=1)
    xray_image_sobel = np.hypot(x_sobel, y_sobel)
    xray_image_sobel *= 255.0 / np.max(xray_image_sobel)
    time_list.append(time.time()-t0)
print(f"Sobel filter takes {sum(time_list)/len(time_list)}s on average")


time_list = []
for _ in range(50):
    t0 = time.time()
    fourier_gaussian = ndimage.fourier_gaussian(xray_image, sigma=0.05)
    x_prewitt = ndimage.prewitt(fourier_gaussian, axis=0)
    y_prewitt = ndimage.prewitt(fourier_gaussian, axis=1)
    xray_image_canny = np.hypot(x_prewitt, y_prewitt)
    xray_image_canny *= 255.0 / np.max(xray_image_canny)
    time_list.append(time.time()-t0)
print(f"Canny filter takes {sum(time_list)/len(time_list)}s on average")
print("==================\n")
