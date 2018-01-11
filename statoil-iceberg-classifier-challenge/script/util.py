# -*- coding: utf-8 -*-

import time
import sys
import os

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.getcwd()
input_path = os.path.join(cwd, 'input')
log_path = os.path.join(cwd, 'log')
model_path = os.path.join(cwd, 'model')
output_path = os.path.join(cwd, 'output')

print('cwd: %s' % cwd)
print('input_path: %s' % input_path)
print('log_path: %s' % log_path)
print('model_path: %s' % model_path)
print('output_path: %s' % output_path)


def get_run_name(project_name, item_name, acc=None):
    date_str = time.strftime("%Y%m%d", time.localtime())
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = project_name + '-' + item_name + '-' + time_str
    if acc != None:
        acc_str = '{0:0>4}'.format(int(acc*10000))
        run_name = run_name + '-' + acc_str
    print('run_name: ' + run_name)
    return run_name

def describe(arr):
    print(arr.shape, arr.min(), arr.max(), sys.getsizeof(arr))

def show_data_images(rows, fig_column, id_data, y_data, *args):
    columns = len(args)
    figs, axes = plt.subplots(rows, columns, figsize=(rows, fig_column*columns))
    print(axes.shape)  
    for i, ax in enumerate(axes):
        y_data_str = ''
        if type(y_data) != type(None):
            y_data_str =  '_' + str(y_data[i])
        ax[0].set_title(id_data[i] + y_data_str)
        for j, arg in enumerate(args):
            ax[j].imshow(arg[i])

def load_sample_submission(is_preview=True):
    sample_submission_path = os.path.join(input_path, 'sample_submission.csv')
    sample_submission = pd.read_csv(sample_submission_path)
    print(sample_submission.shape)
    if is_preview:
        sample_submission.head(2)
    return sample_submission

def load_y_data(is_preview=True):
    is_iceberg_path = os.path.join(input_path, 'is_iceberg.p')
    y_data = pickle.load(open(is_iceberg_path, mode='rb'))
    if is_preview:
        describe(y_data)
    return y_data

def load_data(target_size=75, is_preview=True):
    if target_size == 75:
        target_size_str = ''
    else:
        target_size_str = str(target_size)
    band1_data_path = os.path.join(input_path, 'band1_data_gray%s.p' % target_size_str)
    band2_data_path = os.path.join(input_path, 'band2_data_gray%s.p' % target_size_str)
    band1_test_path = os.path.join(input_path, 'band1_test_gray%s.p' % target_size_str)
    band2_test_path = os.path.join(input_path, 'band2_test_gray%s.p' % target_size_str)

    band1_data = pickle.load(open(band1_data_path, mode='rb'))
    band2_data = pickle.load(open(band2_data_path, mode='rb'))
    band1_test = pickle.load(open(band1_test_path, mode='rb'))
    band2_test = pickle.load(open(band2_test_path, mode='rb'))

    band_max_data = np.maximum(band1_data, band2_data)
    band_max_test = np.maximum(band1_test, band2_test)

    x_data = np.concatenate(
        [band1_data[:, :, :, np.newaxis],
        band2_data[:, :, :, np.newaxis],
        band_max_data[:, :, :, np.newaxis]], axis=-1)

    x_test = np.concatenate(
        [band1_test[:, :, :, np.newaxis],
        band2_test[:, :, :, np.newaxis],
        band_max_test[:, :, :, np.newaxis]], axis=-1)

    if is_preview:
        describe(band1_data)
        describe(band2_data)
        describe(band1_test)
        describe(band2_test)
        describe(band_max_data)
        describe(band_max_test)

        describe(x_data)
        describe(x_test)
    
    return x_data, x_test
