import os


class Config(object):
    # redis conn
    HOST = '18.188.192.187'
    PORT = 6379
    PASSWORD = 'elect'

    # folder
    FEATURE_FOLDER_PATH = os.path.join(os.getcwd(), 'feature')

    # feature
    PRE_TRAINED_MODEL_NAME = 'VGG19'

    # model
    BATCH_SIZE = 1024
    MODEL_FILE = os.path.join(os.getcwd(
    ), 'model', 'Google-LandMark-Rec2019_3-Image-Similar-FCNN-Binary_20190511-164020_8393.h5')
    # test data
    TEST_DATA_NAME = 'test'  # val or test
    TEST_FEATURE_AMOUNT = 1
    REDIS_TEST_IDX_ARR_NAME = 'test_idx_arr'
    REDIS_TEST_PREDICT_RESULT = 'test_pred_result'

    # app sleep
    APP_SLEEP = 3
