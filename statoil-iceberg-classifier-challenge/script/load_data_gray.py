# -*- coding: utf-8 -*-

import time
import sys
import os

import numpy as np
import pandas as pd

from script.util import *

def load_band_data_gray(target_size=75, is_preview=True):
    if target_size == 75:
        target_size_str = ''
    else:
        target_size_str = str(target_size)
    band1_data_gray_file = os.path.join(get_input_processed_folder(), 'band1_data_gray%s.npy' % target_size_str)
    band2_data_gray_file = os.path.join(get_input_processed_folder(), 'band2_data_gray%s.npy' % target_size_str)
    band1_test_gray_file = os.path.join(get_input_processed_folder(), 'band1_test_gray%s.npy' % target_size_str)
    band2_test_gray_file = os.path.join(get_input_processed_folder(), 'band2_test_gray%s.npy' % target_size_str)

    band1_data_gray = np.load(band1_data_gray_file)
    band2_data_gray = np.load(band2_data_gray_file)
    band1_test_gray = np.load(band1_test_gray_file)
    band2_test_gray = np.load(band2_test_gray_file)
    if is_preview:
        describe(band1_data_gray)
        describe(band2_data_gray)
        describe(band1_test_gray)
        describe(band2_test_gray)
    return band1_data_gray, band2_data_gray, band1_test_gray, band2_test_gray

def load_data_gray(target_size=75, is_preview=True):
    if target_size == 75:
        target_size_str = ''
    else:
        target_size_str = str(target_size)
    
    band1_data_gray, band2_data_gray, band1_test_gray, band2_test_gray = load_band_data_gray(target_size=target_size, is_preview=False)
    band_max_data_gray = np.maximum(band1_data_gray, band2_data_gray)
    band_max_test_gray = np.maximum(band1_test_gray, band2_test_gray)

    x_data_gray = np.concatenate(
        [band1_data_gray[:, :, :, np.newaxis],
        band2_data_gray[:, :, :, np.newaxis],
        band_max_data_gray[:, :, :, np.newaxis]], axis=-1)

    x_test_gray = np.concatenate(
        [band1_test_gray[:, :, :, np.newaxis],
        band2_test_gray[:, :, :, np.newaxis],
        band_max_test_gray[:, :, :, np.newaxis]], axis=-1)

    if is_preview:
#         describe(band_max_data)
#         describe(band_max_test)

        describe(x_data_gray)
        describe(x_test_gray)
    
    return x_data_gray, x_test_gray