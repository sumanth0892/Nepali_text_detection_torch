from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def preprocess(img, img_size, data_augmentation=False):
    """
    Convert image into a target image size
    :param img: Original Image
    :param img_size: Size to convert to
    :param data_augmentation: Any augmentations
    :return: Modified image
    """
    if img is None:
        img = np.zeros([img_size[1], img_size[0]])

    w_t, h_t = img_size
    h, w = img.shape
    fx = w / w_t
    fy = h / h_t
    f = max(fx, fy)
    new_size = (max(min(w_t, int(w / f)), 1), max(min(h_t, int(h / f)), 1))

    target = np.ones([h_t, w_t]) * 255
    target[0:new_size[1], 0:new_size[0]] = img

    # Normalize
    m, s = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img -= m
    if s > 0:
        img /= s
    return img

