import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.core.defchararray import not_equal
from scipy.ndimage import gaussian_filter
from skimage.util.dtype import dtype_range
from skimage._shared.utils import check_shape_equality, warn
from skimage.util.arraycrop import crop
from _structural_similarity import stsim


def check_nan(**kwargs):
    arr_name = list(kwargs.keys())[0]
    arr_val = list(kwargs.values())[0]
    nan_nb = np.where(np.isnan(arr_val))[0].shape[0]

    if (nan_nb):
        print(f'{arr_name} contains {nan_nb} nans')

    else:
        print(f'{arr_name} does not contain nan')

# load images
# 1st test => same image
# img1 = img2 = im1 = im2 = cv2.imread("ball1.png")
# img1 = img2 = im1 = im2 = cv2.imread("df1.png")

# 2nd test => slightly different images
# img1 = im1 = cv2.imread("df1.png")
# img2 = im2 = cv2.imread("df2.png")

# 3rd test => completely different images
# img1 = im1 = cv2.imread("df4.png")
# img2 = im2 = cv2.imread("aut1.png")

# 4th test => different ball images (positions)
# img1 = im1 = cv2.imread("ball1.png")
# img2 = im2 = cv2.imread("ball2.png")

# 5th test => different ball images (size)
img1 = im1 = cv2.imread("ball1.png")
img2 = im2 = cv2.imread("ball4.png")

#------------------------------------------------------------
from skimage.metrics import structural_similarity as ssim

# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

k1 = 1e-12
k2 = 1e-12
# s = ssim(im1, im2, K1=k1, K2=k2)
# k1 = k2 = 'default'
s = ssim(im1, im2, multichannel=True, use_sample_covariance=False, gaussian_weights=True, K1=k1, K2=k2)

print(f'For K1 = {k1} and K2 = {k2}\nskimage ssim = {s}')

s = stsim(im1, im2, multichannel=True, use_sample_covariance=False, gaussian_weights=True, K1=k1, K2=k2)

print(f'For K1 = {k1} and K2 = {k2}\nstsim = {s}')