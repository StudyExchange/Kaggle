# -*- coding: utf-8 -*-

import time
import sys
import os

import numpy as np
import pandas as pd

from script.util import *

def load_band_data_gray_sobel(target_size=75, is_preview=True):
    if target_size == 75:
        target_size_str = ''
    else:
        target_size_str = str(target_size)
    band1_data_gray_sobel_file = os.path.join(get_input_processed_folder(), 'band1_data_gray_sobel%s.npy' % target_size_str)
    band2_data_gray_sobel_file = os.path.join(get_input_processed_folder(), 'band2_data_gray_sobel%s.npy' % target_size_str)
    band1_test_gray_sobel_file = os.path.join(get_input_processed_folder(), 'band1_test_gray_sobel%s.npy' % target_size_str)
    band2_test_gray_sobel_file = os.path.join(get_input_processed_folder(), 'band2_test_gray_sobel%s.npy' % target_size_str)

    band1_data_gray_sobel = np.load(band1_data_gray_sobel_file)
    band2_data_gray_sobel = np.load(band2_data_gray_sobel_file)
    band1_test_gray_sobel = np.load(band1_test_gray_sobel_file)
    band2_test_gray_sobel = np.load(band2_test_gray_sobel_file)
    if is_preview:
        describe(band1_data_gray_sobel)
        describe(band2_data_gray_sobel)
        describe(band1_test_gray_sobel)
        describe(band2_test_gray_sobel)
    return band1_data_gray_sobel, band2_data_gray_sobel, band1_test_gray_sobel, band2_test_gray_sobel

def load_data_gray_sobel(target_size=75, is_preview=True):
    if target_size == 75:
        target_size_str = ''
    else:
        target_size_str = str(target_size)
    
    band1_data_gray_sobel, band2_data_gray_sobel, band1_test_gray_sobel, band2_test_gray_sobel = load_band_data_gray_sobel(target_size=target_size, is_preview=False)
    band_max_data_gray_sobel = np.maximum(band1_data_gray_sobel, band2_data_gray_sobel)
    band_max_test_gray_sobel = np.maximum(band1_test_gray_sobel, band2_test_gray_sobel)

    x_data_gray_sobel = np.concatenate(
        [band1_data_gray_sobel[:, :, :, np.newaxis],
        band2_data_gray_sobel[:, :, :, np.newaxis],
        band_max_data_gray_sobel[:, :, :, np.newaxis]], axis=-1)

    x_test_gray_sobel = np.concatenate(
        [band1_test_gray_sobel[:, :, :, np.newaxis],
        band2_test_gray_sobel[:, :, :, np.newaxis],
        band_max_test_gray_sobel[:, :, :, np.newaxis]], axis=-1)

    if is_preview:
#         describe(band_max_data_sobel)
#         describe(band_max_test_sobel)

        describe(x_data_gray_sobel)
        describe(x_test_gray_sobel)
    
    return x_data_gray_sobel, x_test_gray_sobel
