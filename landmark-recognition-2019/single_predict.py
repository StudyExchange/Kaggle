import os
import time
import numpy as np


def get_weight_topn(y_proba, topn, y_data, ndigits=3):
    y_proba_argsorted = [(y_data[item[0]], y_proba[item[0]][0])
                         for item in np.argsort(y_proba, axis=0)[-topn:]]
    class_score_dict = {}
    for item_class, item_proba in y_proba_argsorted:
        if item_class in class_score_dict:
            class_score_dict[item_class] += item_proba
        else:
            class_score_dict[item_class] = item_proba
    class_score_arr = list(class_score_dict.items())
    class_score_arr = sorted(class_score_arr, key=lambda x: x[1])
    topn_pred_arr = [int(item[0]) for item in class_score_arr]
    class_score_arr = [round(float(item[1]), ndigits) for item in class_score_arr]
    return topn_pred_arr, class_score_arr


def get_single_pred(test_idx, main_x, libary_x, libary_y, libary_filename, topn, model, batch_size=1024, ndigits=3):
    t0 = time.time()
    print(len(libary_y))
    print(len(libary_filename))
    # item_main_x = np.array([main_x[test_idx]]*libary_x.shape[0])
    print(main_x.shape)
    # expand_np = np.expand_dims(main_x[test_idx], axis=0)
    print(main_x[test_idx].shape)
    item_main_x = np.tile(main_x[test_idx], (libary_x.shape[0], 1))
    print(item_main_x.shape)
    item_x = {
        'main_input': item_main_x,
        'library_input': libary_x
    }
    y_proba = model.predict(item_x, batch_size=batch_size)
    y_proba_argsort = np.argsort(y_proba, axis=0)
    top1_pred = int(libary_y[np.argmax(y_proba)])
    topn_proba_arr = [round(float(y_proba[item[0]][0]), ndigits)
                      for item in y_proba_argsort[-topn:]]
    topn_pred_arr = [int(libary_y[item[0]])
                     for item in y_proba_argsort[-topn:]]
    topn_filename_arr = [libary_filename[item[0]]
                     for item in y_proba_argsort[-topn:]]
    weighted_topn_pred_arr, weighted_topn_proba_arr = get_weight_topn(
        y_proba, topn, libary_y)
    weighted_top1_pred = weighted_topn_pred_arr[-1]
    time_elapsed = round(time.time() - t0, 2)
    print('%ss' % time_elapsed)
    pred_result = {
        'id': test_idx,
        'top1_pred': top1_pred,
        'topn_pred_arr': topn_pred_arr,
        'topn_proba_arr': topn_proba_arr,
        'topn_filename_arr': topn_filename_arr,
        'weighted_top1_pred': weighted_top1_pred,
        'weighted_topn_pred_arr': weighted_topn_pred_arr,
        'weighted_topn_proba_arr': weighted_topn_proba_arr,
        'time_elapsed': time_elapsed
    }
    return pred_result
