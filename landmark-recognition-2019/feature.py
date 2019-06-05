import os
import h5py
import numpy as np
import pandas as pd

from utility import pickle_load


def get_whole_classes_label(class_indices_data, item_classes_data):
    indices2class = {}
    for i, key in enumerate(class_indices_data.keys()):
        indices2class[class_indices_data[key]] = key
    print(item_classes_data.shape)
    whole_classes_data = np.zeros(item_classes_data.shape)
    for i, class_index in enumerate(item_classes_data):
        whole_classes_data[i] = indices2class[class_index]
    return whole_classes_data


def load_feature(feature_folder_path, data_set_name, model_name, date_str, feature_num_arr):
    x_data_arr = []
    classes_data_arr = []
    filename_data_arr = []
    index_data_arr = []
    for i in feature_num_arr:
        folder_name = 'data_%s_%02d' % (data_set_name, i)
        filenames_file = os.path.join(
            feature_folder_path, 'feature_{0}_{1}_{2}_filenames.pkl'.format(model_name, folder_name, date_str))
        class_indices_file = os.path.join(
            feature_folder_path, 'feature_{0}_{1}_{2}_class_indices.pkl'.format(model_name, folder_name, date_str))
        h5py_file_name = os.path.join(feature_folder_path, 'feature_{0}_{1}_{2}.h5'.format(
            model_name, folder_name, date_str))
        if not os.path.exists(filenames_file):
            print('File not exists', filenames_file)
            continue
        if not os.path.exists(class_indices_file):
            print('File not exists', class_indices_file)
            continue
        if not os.path.exists(h5py_file_name):
            print('File not exists', h5py_file_name)
            continue
        print(filenames_file)
        print(class_indices_file)
        print(h5py_file_name)
        filename_data = pickle_load(filenames_file)
        print('len(filename_data):\t', len(filename_data), type(filename_data))
        class_indices_data = pickle_load(class_indices_file)
        print('len(class_indices_data):\t', len(class_indices_data), type(class_indices_data))

        with h5py.File(h5py_file_name, 'r') as h:
            item_x_data = np.array(h['x_data_%s_%02d' % (data_set_name, i)])
            item_classes_data = np.array(
                h['classes_data_%s_%02d' % (data_set_name, i)])
            item_index_data = np.array(
                h['index_data_%s_%02d' % (data_set_name, i)])
        print(item_x_data.shape)
        x_data_arr.append(item_x_data)
        item_y_data = get_whole_classes_label(
            class_indices_data, item_classes_data)
        print(item_y_data.shape, item_y_data[:10])
        filename_data_arr += filename_data
        classes_data_arr.append(item_y_data)
        index_data_arr.append(item_index_data)
    x_data = np.concatenate(x_data_arr, axis=0)
    y_data = np.concatenate(classes_data_arr, axis=0)
    idx_data = np.concatenate(index_data_arr, axis=0)
    print('x_data.shape:\t', x_data.shape)
    print('y_data.shape:\t', y_data.shape)
    print('len(filename_data_arr):\t', len(filename_data_arr))
    print('idx_data.shape:\t', idx_data.shape)
    print('*' * 60)
    return x_data, y_data, filename_data_arr, idx_data
