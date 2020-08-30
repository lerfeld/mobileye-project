import os
import json
import glob
import argparse
import random

import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
from scipy import signal as sg
import scipy.ndimage as ndimage
from scipy.ndimage.filters import maximum_filter
#from skimage.feature import peak_local_max
from PIL import Image
import matplotlib.pyplot as plt


images_url = '../../data/leftImg8bit_trainvaltest/leftImg8bit/train/aachen'
images_list = glob.glob(os.path.join(images_url, '*_leftImg8bit.png'))
labelIds_url = '../../data/gtFine_trainvaltest/gtFine/train/aachen'
labelIds_list = glob.glob(os.path.join(labelIds_url, "*_labelIds.png"))


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def crop(image_, indexes):
    index = random.randrange(len(indexes[0]))
    x = indexes[0][index]
    y = indexes[1][index]
    crop_image = image_[x:x + 82, y:y + 82]
    plt.imshow(crop_image)
    plt.show(block=True)


for lable, image in zip(labelIds_list, images_list):
    img = np.array(Image.open(lable))
    index_of_tfl = np.where(img == 19)
    index_of_not_tfl = np.where(img != 19)

    original_img = np.array(Image.open(image))
    big_original_img = np.pad(original_img, 40, pad_with)[:, :, 40:43]

    if(index_of_tfl[0] != []):
        crop(big_original_img, index_of_tfl)
        crop(big_original_img, index_of_not_tfl)
