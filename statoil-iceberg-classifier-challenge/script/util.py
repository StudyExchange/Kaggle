# -*- coding: utf-8 -*-

import time
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_run_name(project_name, item_name, acc=None):
    date_str = time.strftime("%Y%m%d", time.localtime())
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = project_name + '-' + item_name + '-' + time_str
    if acc != None:
        acc_str = '{0:0>4}'.format(int(acc*10000))
        run_name = run_name + '-' + acc_str
    print('run_name: ' + run_name)
    return run_name

def get_html_folder(is_preview=False):
    html_folder = os.path.join(os.getcwd(), 'html')
    if not os.path.exists(html_folder):
        os.mkdir(html_folder)
    if is_preview:
        print(html_folder)
    return html_folder

def get_input_folder(is_preview=False):
    input_folder = os.path.join(os.getcwd(), 'input')
    if not os.path.exists(input_folder):
        os.mkdir(input_folder)
    if is_preview:
        print(input_folder)
    return input_folder

def get_input_processed_folder(is_preview=False):
    input_processed_folder = os.path.join(os.getcwd(), 'input', 'processed')
    if not os.path.exists(input_processed_folder):
        os.mkdir(input_processed_folder)
    if is_preview:
        print(input_processed_folder)
    return input_processed_folder

def get_log_folder(is_preview=False):
    log_folder = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if is_preview:
        print(log_folder)
    return log_folder

def get_model_folder(is_preview=False):
    model_folder = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    if is_preview:
        print(model_folder)
    return model_folder

def get_output_folder(is_preview=False):
    output_folder = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if is_preview:
        print(output_folder)
    return output_folder

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
    sample_submission_file = os.path.join(get_input_folder(), 'sample_submission.csv')
    sample_submission = pd.read_csv(sample_submission_file)
    print(sample_submission.shape)
    if is_preview:
        print(sample_submission.head(2))
    return sample_submission

def load_id(is_preview=True):
    id_data_file = os.path.join(get_input_processed_folder(), 'id_data.npy')
    id_test_file = os.path.join(get_input_processed_folder(), 'id_test.npy')
    id_data = np.load(id_data_file)
    id_test = np.load(id_test_file)
    if is_preview:
        describe(id_data)
        describe(id_test)
    return id_data, id_test

def load_y_data(is_preview=True):
    y_data_file = os.path.join(get_input_processed_folder(), 'y_data.npy')
    y_data = np.load(y_data_file)
    if is_preview:
        describe(y_data)
    return y_data

def load_inc_angle_data(is_preview=True):
    inc_angle_data_file = os.path.join(get_input_processed_folder(), 'inc_angle_data.npy')
    inc_angle_test_file = os.path.join(get_input_processed_folder(), 'inc_angle_test.npy')
    inc_angle_data = np.load(inc_angle_data_file)
    inc_angle_test = np.load(inc_angle_test_file)
    if is_preview:
        describe(inc_angle_data)
        describe(inc_angle_test)
    return inc_angle_data, inc_angle_test

def load_band_data(target_size=75, is_preview=True):
    if target_size == 75:
        target_size_str = ''
    else:
        target_size_str = str(target_size)
    band1_data_file = os.path.join(get_input_processed_folder(), 'band1_data%s.npy' % target_size_str)
    band2_data_file = os.path.join(get_input_processed_folder(), 'band2_data%s.npy' % target_size_str)
    band1_test_file = os.path.join(get_input_processed_folder(), 'band1_test%s.npy' % target_size_str)
    band2_test_file = os.path.join(get_input_processed_folder(), 'band2_test%s.npy' % target_size_str)

    band1_data = np.load(band1_data_file)
    band2_data = np.load(band2_data_file)
    band1_test = np.load(band1_test_file)
    band2_test = np.load(band2_test_file)
    if is_preview:
        describe(band1_data)
        describe(band2_data)
        describe(band1_test)
        describe(band2_test)
    return band1_data, band2_data, band1_test, band2_test

def load_data(target_size=75, is_preview=True):
    if target_size == 75:
        target_size_str = ''
    else:
        target_size_str = str(target_size)
    
    band1_data, band2_data, band1_test, band2_test = load_band_data(target_size=target_size, is_preview=False)
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
#         describe(band_max_data)
#         describe(band_max_test)

        describe(x_data)
        describe(x_test)
    
    return x_data, x_test
