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

def custom_ssim(img1, img2, *, multichannel=True, **kwargs):
    check_shape_equality(img1, img2)

    if multichannel:
        # loop over channels
        args = dict(multichannel=False)
        args.update(kwargs)
        nch = img1.shape[-1]
        mssim = np.empty(nch)

        for ch in range(nch):
            ch_result = custom_ssim(im1[..., ch],
                                              im2[..., ch], **args)
            mssim[..., ch] = ch_result
        mssim = mssim.mean()
        return mssim
    
    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)

    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1

    if img1.dtype != img2.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "img1.dtype.", stacklevel=2)
    dmin, dmax = dtype_range[img1.dtype.type]
    data_range = dmax - dmin

    ndim = img1.ndim

    filter_args = {'sigma': sigma, 'truncate': truncate}

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # compute (weighted) means
    ux = gaussian_filter(img1, **filter_args)
    uy = gaussian_filter(img2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = gaussian_filter(img1 * img1, **filter_args)
    uyy = gaussian_filter(img2 * img2, **filter_args)
    uxy = gaussian_filter(img1 * img2, **filter_args)
    print(np.mean(ux ** 2 + uy ** 2))
    vx = (uxx - ux * ux)
    vy = (uyy - uy * uy)
    vxy = (uxy - ux * uy)
    print(np.mean(vx + vy))

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))

    # guarantee that there is no NaN in S
    S1 = A1 / B1
    S2 = A2 / B2
    
    S1[B1==0.0] = 1.0
    S2[B2==0.0] = 1.0

    S = S1 * S2

    

    print(f'min(A1 * A2) = {np.min(A1 * A2)} | max(A1 * A2) = {np.max(A1 * A2)}')
    
    check_nan(numerator = A1 * A2)
    check_nan(S = S)

    print(f'S = {S[300]}')

    # plt.figure()
    # plt.imshow(S, cmap=plt.cm.gray)
    # plt.title(S)
    # plt.figure()
    # plt.imshow(ux, cmap=plt.cm.gray)
    # plt.title(ux)

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()
    # print(f'ux= {ux}')
    print(f'ux = {np.mean(ux)}, uy = {np.mean(uy)},\
        uxx = {np.mean(uxx)}, uyy = {np.mean(uyy)},\
            uxy = {np.mean(uxy)}')

    return mssim

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

# img1 = im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cs = custom_ssim(img1, img2, K1=0, K2=0)

print(f'custom ssim = {cs.mean()}')

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

# # Effect of Gaussian Filter

# im1_gauss = gaussian_filter(im1, sigma=1.5, truncate = 3.5)
# im_list = [im1, im1_gauss]
# im_title_list = ['Original', 'Gaussian filter']

# fig = plt.figure('Effect of Gaussian Filter', figsize=(8, 5))
# columns = 2
# rows = 1
# for i in range(0, columns*rows):
#     fig.add_subplot(rows, columns, i+1)
#     im_list[i] = cv2.cvtColor(im_list[i], cv2.COLOR_BGR2GRAY)
#     plt.imshow(im_list[i], cmap=plt.cm.gray)
#     plt.title(im_title_list[i])
# plt.show()