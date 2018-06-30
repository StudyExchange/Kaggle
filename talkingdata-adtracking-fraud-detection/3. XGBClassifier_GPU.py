
# coding: utf-8

# # 3. XGBClassifier_GPU
# **Start from the most basic features, and try to improve step by step.**
# Reference:
# - XGBoost Parameters, http://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
# - Python API Reference, http://xgboost.readthedocs.io/en/latest/python/python_api.html
# - https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# ## Run name

# In[1]:


import time

project_name = 'TalkingdataAFD2018'
step_name = 'XGBClassifier_GPU'
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
run_name = '%s_%s_%s' % (project_name, step_name, time_str)
print('run_name: %s' % run_name)
t0 = time.time()


# ## Important params

# In[ ]:


date = 6
print('date: ', date)

test_n_rows = None
# test_n_rows = 18790469
# test_n_rows = 10*10000


# In[ ]:


day_rows = {
    0: {
        'n_skiprows': 1,
        'n_rows': 10 * 10000
    },
    6: {
        'n_skiprows': 1,
        'n_rows': 9308568
    },
    7: {
        'n_skiprows': 1 + 9308568,
        'n_rows': 59633310
    },
    8: {
        'n_skiprows': 1 + 9308568 + 59633310,
        'n_rows': 62945075
    },
    9: {
        'n_skiprows': 1 + 9308568 + 59633310 + 62945075,
        'n_rows': 53016937
    }
}
n_skiprows = day_rows[date]['n_skiprows']
n_rows = day_rows[date]['n_rows']


# ## Import PKGs

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display

import os
import gc
import time
import random
import zipfile
import h5py
import pickle
import math
from PIL import Image
import shutil

from tqdm import tqdm
import multiprocessing
from multiprocessing import cpu_count

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

random_num = np.random.randint(10000)
print('random_num: %s' % random_num)


# ## Project folders

# In[ ]:


cwd = os.getcwd()

input_folder = os.path.join(cwd, 'input')
output_folder = os.path.join(cwd, 'output')
model_folder = os.path.join(cwd, 'model')
log_folder = os.path.join(cwd, 'log')
print('input_folder: \t\t\t%s' % input_folder)
print('output_folder: \t\t\t%s' % output_folder)
print('model_folder: \t\t\t%s' % model_folder)
print('log_folder: \t\t\t%s' % log_folder)

train_csv_file = os.path.join(input_folder, 'train.csv')
train_sample_csv_file = os.path.join(input_folder, 'train_sample.csv')
test_csv_file = os.path.join(input_folder, 'test.csv')
sample_submission_csv_file = os.path.join(input_folder, 'sample_submission.csv')

print('\ntrain_csv_file: \t\t%s' % train_csv_file)
print('train_sample_csv_file: \t\t%s' % train_sample_csv_file)
print('test_csv_file: \t\t\t%s' % test_csv_file)
print('sample_submission_csv_file: \t%s' % sample_submission_csv_file)


# ## Load data

# In[ ]:


# %%time

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

train_csv = pd.read_csv(
    train_csv_file, 
    skiprows=range(1, n_skiprows), 
    nrows=n_rows, 
    usecols=train_columns,
    dtype=dtypes,
    parse_dates=['click_time']
)
test_csv = pd.read_csv(
    test_csv_file, 
    nrows=test_n_rows, 
    usecols=test_columns,
    dtype=dtypes,
    parse_dates=['click_time']
)
sample_submission_csv = pd.read_csv(sample_submission_csv_file)

print('train_csv.shape: \t\t', train_csv.shape)
print('test_csv.shape: \t\t', test_csv.shape)
print('sample_submission_csv.shape: \t', sample_submission_csv.shape)
print('train_csv.dtypes: \n', train_csv.dtypes)

display(train_csv.head(2))
display(test_csv.head(2))
display(sample_submission_csv.head(2))


# In[ ]:


y_data = train_csv['is_attributed']
train_csv.drop(['is_attributed'], axis=1, inplace=True)
display(y_data.head())


# ## Features

# In[ ]:


train_csv['day'] = train_csv['click_time'].dt.day.astype('uint8')
train_csv['hour'] = train_csv['click_time'].dt.hour.astype('uint8')
train_csv['minute'] = train_csv['click_time'].dt.minute.astype('uint8')
train_csv['second'] = train_csv['click_time'].dt.second.astype('uint8')
print('train_csv.shape: \t', train_csv.shape)
display(train_csv.head(2))


# In[ ]:


test_csv['day'] = test_csv['click_time'].dt.day.astype('uint8')
test_csv['hour'] = test_csv['click_time'].dt.hour.astype('uint8')
test_csv['minute'] = test_csv['click_time'].dt.minute.astype('uint8')
test_csv['second'] = test_csv['click_time'].dt.second.astype('uint8')
print('test_csv.shape: \t', test_csv.shape)
display(test_csv.head(2))


# In[ ]:


arr = np.array([[3,6,6],[4,5,1]])
print(arr)
np.ravel_multi_index(arr, (7,6))
print(arr)
print(np.ravel_multi_index(arr, (7,6), order='F'))


# In[ ]:


def df_add_counts(df, cols, tag="_count"):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(
        np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1), 
        return_inverse=True, 
        return_counts=True
    )
    df["_".join(cols) + tag] = counts[unqtags]
    return df


# In[ ]:


def df_add_uniques(df, cols, tag="_unique"):
    gp = df[cols]         .groupby(by=cols[0:len(cols) - 1])[cols[len(cols) - 1]]         .nunique()         .reset_index()         .rename(index=str, columns={cols[len(cols) - 1]: "_".join(cols)+tag})
    df = df.merge(gp, on=cols[0:len(cols) - 1], how='left')
    return df


# In[ ]:


train_csv = df_add_counts(train_csv, ['ip', 'day', 'hour'])
train_csv = df_add_counts(train_csv, ['ip', 'app'])
train_csv = df_add_counts(train_csv, ['ip', 'app', 'os'])
train_csv = df_add_counts(train_csv, ['ip', 'device'])
train_csv = df_add_counts(train_csv, ['app', 'channel'])
train_csv = df_add_uniques(train_csv, ['ip', 'channel'])
train_csv = df_add_uniques(train_csv, ['ip', 'app', 'device', 'os', 'channel'])
train_csv = df_add_uniques(train_csv, ['ip', 'os', 'device'])
train_csv = df_add_uniques(train_csv, ['ip', 'os', 'device', 'app'])

display(train_csv.head())


# In[ ]:


test_csv = df_add_counts(test_csv, ['ip', 'day', 'hour'])
test_csv = df_add_counts(test_csv, ['ip', 'app'])
test_csv = df_add_counts(test_csv, ['ip', 'app', 'os'])
test_csv = df_add_counts(test_csv, ['ip', 'device'])
test_csv = df_add_counts(test_csv, ['app', 'channel'])
test_csv = df_add_uniques(test_csv, ['ip', 'channel'])
test_csv = df_add_uniques(test_csv, ['ip', 'app', 'device', 'os', 'channel'])
test_csv = df_add_uniques(test_csv, ['ip', 'os', 'device'])
test_csv = df_add_uniques(test_csv, ['ip', 'os', 'device', 'app'])

display(test_csv.head())


# ## Prepare data

# In[ ]:


train_useless_features = ['click_time']
train_csv.drop(train_useless_features, axis=1, inplace=True)

test_useless_features = ['click_time', 'click_id']
test_csv.drop(test_useless_features, axis=1, inplace=True)

display(train_csv.head())
display(test_csv.head())


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(train_csv, y_data, test_size=0.01, random_state=2017)
x_test = test_csv
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)

print('Time cost: %.2f s' % (time.time() - t0))


# ## Train

# In[ ]:


# %%time

import xgboost as xgb
from sklearn.metrics import roc_auc_score


clf = xgb.XGBClassifier(
    max_depth=100, 
    learning_rate=0.1, 
    n_estimators=1000, 
    silent=False, 
    objective='gpu:binary:logistic', 
    booster='gbtree', 
    n_jobs=1, 
    nthread=None, 
    gamma=0, 
    min_child_weight=1, 
    max_delta_step=0, 
    subsample=0.7, 
    colsample_bytree=0.7, 
#     colsample_bylevel=0.7, 
    reg_alpha=0.01, 
    reg_lambda=0.99, 
    scale_pos_weight=97, 
    base_score=0.5, 
    random_state=random_num, 
    seed=None, 
    missing=None,
    # booster params
    num_boost_round=50,
#     early_stopping_rounds=10,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
#     eval_metric=['auc', 'logloss']
)

clf.fit(
    x_train, 
    y_train,
#     sample_weight=None, 
    eval_set=[(x_train, y_train), (x_val, y_val)], 
    eval_metric=['auc', 'logloss', 'error'], 
    early_stopping_rounds=20, 
#     verbose=False, 
#     xgb_model=None
)


print('*' * 80)
y_train_proba = clf.predict(x_train)
y_train_pred = (y_train_proba>=0.5).astype(int)
acc_train = accuracy_score(y_train, y_train_pred)
roc_train = roc_auc_score(y_train, y_train_proba)
print('acc_train: %.4f \t roc_train: %.4f' % (acc_train, roc_train))

y_val_proba = clf.predict(x_val)
y_val_pred = (y_val_proba>=0.5).astype(int)
acc_val = accuracy_score(y_val, y_val_pred)
roc_val = roc_auc_score(y_val, y_val_proba)
print('acc_val:   %.4f \t roc_val:   %.4f' % (acc_val, roc_val))


# In[ ]:


evals_result = clf.evals_result()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

n_round = len(evals_result['validation_0']['auc'])
x = list(range(1, n_round + 1))
for metric_key in evals_result['validation_0'].keys():
    plt.xlabel('echo')
    plt.ylabel(metric_key)
    for i, val_key in enumerate(evals_result.keys()):
        plt.plot(x, evals_result[val_key][metric_key], colors[i])
    plt.legend(labels = list(evals_result.keys()), loc = 'best')
    plt.show()


# ## Predict

# In[ ]:


run_name_acc = run_name + '_' + str(int(roc_val*10000)).zfill(4)
print(run_name_acc)


# In[ ]:


y_test_proba = clf.predict(x_test)
print(y_test_proba.shape)
print(y_test_proba[:20])


# In[ ]:


def save_proba(y_train_proba, y_train, y_val_proba, y_val, y_test_proba, click_ids, file_name):
    print(click_ids[:5])
    if os.path.exists(file_name):
        os.remove(file_name)
        print('File removed: \t%s' % file_name)
    with h5py.File(file_name) as h:
        h.create_dataset('y_train_proba', data=y_train_proba)
        h.create_dataset('y_train', data=y_train)
        h.create_dataset('y_val_proba', data=y_val_proba)
        h.create_dataset('y_val', data=y_val)
        h.create_dataset('y_test_proba', data=y_test_proba)
        h.create_dataset('click_ids', data=click_ids)
    print('File saved: \t%s' % file_name)

def load_proba(file_name):
    with h5py.File(file_name, 'r') as h:
        y_train_proba = np.array(h['y_train_proba'])
        y_train = np.array(h['y_train'])
        y_val_proba = np.array(h['y_val_proba'])
        y_val = np.array(h['y_val'])
        y_test_proba = np.array(h['y_test_proba'])
        click_ids = np.array(h['click_ids'])
    print('File loaded: \t%s' % file_name)
    print(click_ids[:5])
    
    return y_train_proba, y_train, y_val_proba, y_val, y_test_proba, click_ids


y_proba_file = os.path.join(model_folder, 'proba_%s.p' % run_name_acc)
save_proba(y_train_proba, y_train, y_val_proba, y_val, y_test_proba, np.array(sample_submission_csv['click_id']), y_proba_file)
y_train_proba, y_train, y_val_proba, y_val, y_test_proba, click_ids = load_proba(y_proba_file)

print(y_train_proba.shape)
print(y_train.shape)
print(y_val_proba.shape)
print(y_val.shape)
print(y_test_proba.shape)
print(len(click_ids))


# In[ ]:


# %%time
submission_csv_file = os.path.join(output_folder, 'pred_%s.csv' % run_name_acc)
print(submission_csv_file)
submission_csv = pd.DataFrame({ 'click_id': click_ids , 'is_attributed': y_test_proba })
submission_csv.to_csv(submission_csv_file, index = False)


# In[ ]:


print('Time cost: %.2f s' % (time.time() - t0))

print('random_num: ', random_num)
print('date: ', date)
print(run_name_acc)
print('Done!')

