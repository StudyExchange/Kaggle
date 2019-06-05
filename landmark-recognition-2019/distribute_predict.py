import os
import json
import time
import redis
import platform
platform_name = platform.platform().lower()
is_win = 'windows' in platform_name

# tensorflow
import tensorflow as tf

if not is_win:
    import keras.backend.tensorflow_backend as KTF

# Keras
import keras
from keras.utils import Sequence
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.preprocessing.image import *
from keras.applications.inception_v3 import preprocess_input

# my pkg
from config import Config
from utility import pickle_dump, pickle_load
from task import get_one_task
from feature import load_feature
from test_data import get_one_test_image_index
from predict_result import save_predict_result
from single_predict import get_single_pred


def run_task(x_test, x_train, y_train, filename_train, topn, model, redis_cli):
    # 1. Fetch one image idx from redis
    test_image_idx = get_one_test_image_index(
        Config.REDIS_TEST_IDX_ARR_NAME, redis_cli)
    if test_image_idx is None or test_image_idx >= x_test.shape[0]:
        print('App sleep %ds' % Config.APP_SLEEP)
        time.sleep(Config.APP_SLEEP)
        return
    # 2. Predict
    pred_result = get_single_pred(
        test_image_idx, x_test, x_train, y_train, filename_train, topn, model, Config.BATCH_SIZE)
    print(type(test_image_idx), test_image_idx, pred_result['id'], pred_result['top1_pred'], len(pred_result['topn_pred_arr']), pred_result['weighted_top1_pred'], len(pred_result['weighted_topn_pred_arr']))
    # 3. Save result to redis
    save_predict_result(
        pred_result, Config.REDIS_TEST_PREDICT_RESULT, redis_cli)


def main():
    # Init redis connection
    redis_cli = redis.Redis(
        host=Config.HOST, port=Config.PORT, password=Config.PASSWORD)
    # Fetch task params from redis
    task = get_one_task(redis_cli)
    if not task:
        print('Do not have task, APP exit.')
    print(task)
    # # Init tensorflow runtime params
    if not is_win:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = task['per_process_gpu_memory_fraction']
        tf_session = tf.Session(config=tf_config)
        KTF.set_session(tf_session)
    # Load model
    model_file = os.path.join(Config.MODEL_FILE)
    print(model_file)
    model = load_model(model_file)
    print(model.summary())
    # Load feature
    x_train, y_train, filename_train, _ = load_feature(
        Config.FEATURE_FOLDER_PATH, 'train', Config.PRE_TRAINED_MODEL_NAME, task['model_date_str'], list(range(task['libary_batch_amount'])))
    x_test, _, _, _ = load_feature(
        Config.FEATURE_FOLDER_PATH, Config.TEST_DATA_NAME, Config.PRE_TRAINED_MODEL_NAME, task['model_date_str'], [1])

    # Run task
    while(1):
        run_task(x_test, x_train, y_train, filename_train, task['topn'], model, redis_cli)


if __name__ == "__main__":
    print('App start!')
    main()
    print('App finished!')
