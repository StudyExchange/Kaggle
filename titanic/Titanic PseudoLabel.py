
# coding: utf-8

# # Titanic PseudoLabel
# 
# Kaggle score: 
# 
# 重要：
# - 因为model.fit(features.as_matrix(), survived.as_matrix(), batch_size = 2, epochs = 20)需要numpy.array输入，而不是pandas.DataFrame，这里需要DataFrame.as_matrix()转换
# - 因为使用了kernel_initializer = 'uniform'，导致报错：InternalError: Blas GEMM launch failed
# 
# Reference: 
# 1. https://www.kaggle.com/c/titanic#tutorials
# 2. https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# 3. https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook
# 

# ### Import pkgs

# In[1]:


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


# ## Run name

# In[2]:


project_name = 'Titanic'
step_name = 'PseudoLabel'
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
run_name = project_name + '_' + step_name + '_' + time_str
print('run_name: ' + run_name)


# ## Project folders

# In[3]:


cwd = os.getcwd()
date_str = '20180409_0040'
input_folder = os.path.join(cwd, 'input')
output_folder = os.path.join(cwd, 'output')
output_temp_folder = os.path.join(cwd, 'output', date_str)
model_folder = os.path.join(cwd, 'model')
model_temp_folder = os.path.join(cwd, 'model', date_str)
feature_folder = os.path.join(cwd, 'feature')
log_folder = os.path.join(cwd, 'log')
print('input_folder: \t\t\t%s' % input_folder)
print('output_folder: \t\t\t%s' % output_folder)
print('output_temp_folder: \t\t%s' % output_temp_folder)
print('model_folder: \t\t\t%s' % model_folder)
print('model_temp_folder: \t\t%s' % model_temp_folder)
print('feature_folder: \t\t%s' % feature_folder)
print('log_folder: \t\t\t%s' % log_folder)

if not os.path.exists(output_temp_folder):
    os.mkdir(output_temp_folder)
    print('Create folder: %s' % output_temp_folder)
if not os.path.exists(model_temp_folder):
    os.mkdir(model_temp_folder)
    print('Create folder: %s' % model_temp_folder)

train_csv_file = os.path.join(input_folder, 'train.csv')
test_csv_file = os.path.join(input_folder, 'test.csv')

print(train_csv_file)
print(test_csv_file)


# ### Import original data as DataFrame

# In[4]:


data_train = pd.read_csv(train_csv_file)
data_test = pd.read_csv(test_csv_file)

display(data_train.head(20))
display(data_test.head(20))
data_train.loc[2, 'Ticket']


# ### Show columns of dataframe

# In[5]:


data_train_original_col = data_train.columns
data_test_original_col = data_test.columns
print(data_train_original_col)
print(data_test_original_col)
# data_train0 = data_train.drop(data_train_original_col, axis = 1)
# data_test0  = data_test.drop(data_test_original_col, axis = 1)
# display(data_train0.head(2))
# display(data_test0.head(2))


# ### Preprocess features

# In[6]:


full_data = [data_train, data_test]


# In[7]:


# Pclass
for dataset in full_data:
    temp = dataset[dataset['Pclass'].isnull()]
    if len(temp) == 0:
        print('Do not have null value!')
    else:
        temp.head(2)
        
for dataset in full_data:
    dataset['a_Pclass'] = dataset['Pclass']
#     display(dataset.head())


# In[8]:


# Name
for dataset in full_data:
    dataset['a_Name_Length'] = dataset['Name'].apply(len)
#     display(dataset.head(2))


# In[9]:


# Sex
for dataset in full_data:
    dataset['a_Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
#     display(dataset.head(2))


# In[10]:


# Age
for dataset in full_data:
    dataset['a_Age'] = dataset['Age'].fillna(-1)
    dataset['a_Have_Age'] = dataset['Age'].isnull().map({True: 0, False: 1}).astype(int)
#     display(dataset[dataset['Age'].isnull()].head(2))
#     display(dataset.head(2))


# In[11]:


# SibSp and Parch
for dataset in full_data:
    dataset['a_FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['a_IsAlone'] = dataset['a_FamilySize'].apply(lambda x: 1 if x<=1 else 0)
#     display(dataset.head(2))


# In[12]:


# Ticket(Very one have a ticket)
for dataset in full_data:
    dataset['a_Have_Ticket'] = dataset['Ticket'].isnull().map({True: 0, False: 1}).astype(int)
#     display(dataset[dataset['Ticket'].isnull()].head(2))
#     display(dataset.head(2))


# In[13]:


# Fare
for dataset in full_data:
    dataset['a_Fare'] = dataset['Fare'].fillna(-1)
    dataset['a_Have_Fare'] = dataset['Fare'].isnull().map({True: 0, False: 1}).astype(int)
#     display(dataset[dataset['Fare'].isnull()].head(2))
#     display(dataset.head(2))


# In[14]:


# Cabin
for dataset in full_data:
    dataset['a_Have_Cabin'] = dataset['Cabin'].isnull().map({True: 0, False: 1}).astype(int)
#     display(dataset[dataset['Cabin'].isnull()].head(2))
#     display(dataset.head(2))


# In[15]:


# Embarked
for dataset in full_data:
#     dataset['Embarked'] = dataset['Embarked'].fillna('N')
    dataset['a_Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, None: 3} ).astype(int)
    dataset['a_Have_Embarked'] = dataset['Embarked'].isnull().map({True: 0, False: 1}).astype(int)
#     display(dataset[dataset['Embarked'].isnull()].head(2))
#     display(dataset.head(2))


# Name words segmentation and one-hote

# In[16]:


# Name words segmentation
import re
name_words = []

# Inorder to allign columns of data_train and data_test, only data_train to fetch word
for name in data_train['Name']:
#     print(name)
    words = re.findall(r"[\w']+", name)
#     print(len(words))
#     print(words)
    for w in words:
        if w not in name_words:
            name_words.append(w)
# print(len(name_words))
name_words.sort()
# print(name_words)


# In[17]:


# Add columns
for dataset in full_data:
    for w in name_words:
        col_name = 'a_Name_' + w
        dataset[col_name] = 0
    dataset.head(1)


# In[18]:


# Name words one-hote
for dataset in full_data:
    for i, row in dataset.iterrows():
    #     print(row['Name'])
        words = re.findall(r"[\w']+", row['Name'])
        for w in words:
            if w in name_words:
                col_name = 'a_Name_' + w
                dataset.loc[i, col_name] = 1
#     display(dataset[dataset['a_Name_Braund'] == 1])


# Cabin segmentation and one-hote

# In[19]:


# Get cabin segmentation words
import re
cabin_words = []

# Inorder to allign columns of data_train and data_test, only data_train to fetch number
for c in data_train['Cabin']:
#     print(c)
    if c is not np.nan:
        word = re.findall(r"[a-zA-Z]", c)
#         print(words[0])
        cabin_words.append(word[0])
print(len(cabin_words))
cabin_words.sort()
print(np.unique(cabin_words))
cabin_words_unique = list(np.unique(cabin_words))


# In[20]:


def get_cabin_word(cabin):
    if cabin is not np.nan:
        word = re.findall(r"[a-zA-Z]", cabin)
        if word:
            return cabin_words_unique.index(word[0])
    return -1

for dataset in full_data:
    dataset['a_Cabin_Word'] = dataset['Cabin'].apply(get_cabin_word)
    # dataset['a_Cabin_Word'].head(100)


# In[21]:


def get_cabin_number(cabin):
    if cabin is not np.nan:
        word = re.findall(r"[0-9]+", cabin)
        if word:
            return int(word[0])
    return -1

for dataset in full_data:
    dataset['a_Cabin_Number'] = dataset['Cabin'].apply(get_cabin_number)
    # dataset['a_Cabin_Number'].head(100)


# In[22]:


# Clean data
# Reference: 
#    1. https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
#    2. https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook

full_data = [data_train, data_test]
for dataset in full_data:
    dataset['a_Name_length'] = dataset['Name'].apply(len)
    #dataset['Sex'] = (dataset['Sex']=='male').astype(int)
    dataset['a_Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    dataset['a_Age'] = dataset['Age'].fillna(0)
    dataset['a_Age_IsNull'] = dataset['Age'].isnull()
    dataset['a_FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['a_IsAlone'] = dataset['a_FamilySize'].apply(lambda x: 1 if x<=1 else 0)
    dataset['a_Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    #dataset['Has_Cabin'] = dataset['Cabin'].apply(lambda x: 1 if type(x) == str else 0) # same as below
    dataset['a_Has_Cabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    dataset['a_Has_Embarked'] = dataset['Embarked'].isnull()
    dataset['Embarked'] = dataset['Embarked'].fillna('N')
    dataset['a_Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'N': 3} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
display(data_train.head(2))
display(data_test.head(2))


# In[23]:


survived = data_train['Survived']
data_train0 = data_train.drop(data_train_original_col, axis = 1)
data_test0  = data_test.drop(data_test_original_col, axis = 1)
display(data_train0.head(2))
display(data_test0.head(2))

features = data_train0
display(features.head(2))


# Check and confirm all columns is proccessed

# In[24]:


for col in features.columns:
    if not col.startswith('a_'):
        print(col)


# ## 2. Build model

# In[25]:


from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

x_data = features
y_data = survived
x_test = data_test0

n_components = random.choice([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
print('n_components: %s' % n_components)

pca = PCA(n_components=n_components)
pca.fit(x_data)
pca.fit(x_test)

x_data = pca.transform(x_data)
x_test = pca.transform(x_test)

random_num = np.random.randint(10000)
print('random_num: %s' % random_num)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=random_num)
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)
print(x_test.shape)


lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

# LightGBM parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
#         'num_class': 3,
    'learning_rate': random.choice([0.03, 0.1]),
    'num_leaves': random.choice([5, 10, 20, 30, 40]),
    'max_depth': random.choice([4, 5, 6, 7, 8, 9, 10, 11, 12]),
    'n_estimators': random.choice([2000, 5000, 10000]),
    'min_data_in_leaf': random.choice([5, 10, 20, 30, 40]),
    'num_iteration': random.choice([30, 40, 60, 80]),
    'verbose': 0
}

print('params: %s' % params)

# train
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=lgb_eval,
    early_stopping_rounds=10
)

y_val_prob = gbm.predict(x_val, num_iteration=gbm.best_iteration)
print(y_val_prob.shape)
print(y_val_prob[:10])
val_pred_test = (y_val_prob>=0.5).astype(int)
print(val_pred_test[:10])
val_acc = accuracy_score(val_pred_test, y_val)
print('val_acc: %.3f' % val_acc)
print('*' * 60)

y_test_proba = gbm.predict(x_test, num_iteration=gbm.best_iteration)
# y_pred = np.argmax(y_pred, axis=1)
print(y_test_proba.shape)
print(y_test_proba[:10])
y_test_pred = (y_test_proba>=0.5).astype(int)
print(y_test_pred[:10])

y_data_proba = gbm.predict(x_data, num_iteration=gbm.best_iteration)
# y_pred = np.argmax(y_pred, axis=1)
print(y_data_proba.shape)
print(y_data_proba[:10])
y_data_pred = (y_data_proba>=0.5).astype(int)
print(y_data_pred[:10])


# In[26]:


print('random_num: %s' % random_num)
print('val_acc: %.3f' % val_acc)


# ## Pseudo label

# In[27]:


print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test_pred.shape)


# In[28]:


from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

ps_x_data = [x_train, x_val, x_test[:200]]
ps_y_data = [y_train, y_val, y_test_pred[:200]]
ps_x_data = np.concatenate(ps_x_data, axis=0)
ps_y_data = np.concatenate(ps_y_data, axis=0)
print(ps_x_data.shape)
print(ps_y_data.shape)


lgb_train = lgb.Dataset(ps_x_data, ps_y_data)
lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

# LightGBM parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
#         'num_class': 3,
    'learning_rate': random.choice([0.03, 0.1]),
    'num_leaves': random.choice([5, 10, 20, 30, 40]),
    'max_depth': random.choice([4, 5, 6, 7, 8, 9, 10, 11, 12]),
    'n_estimators': random.choice([2000, 5000, 10000]),
    'min_data_in_leaf': random.choice([5, 10, 20, 30, 40]),
    'num_iteration': random.choice([30, 40, 60, 80]),
    'verbose': 0
}

print('params: %s' % params)

# train
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=lgb_eval,
    early_stopping_rounds=10
)

y_val_prob = gbm.predict(x_val, num_iteration=gbm.best_iteration)
print(y_val_prob.shape)
print(y_val_prob[:10])
val_pred_test = (y_val_prob>=0.5).astype(int)
print(val_pred_test[:10])
val_acc1 = accuracy_score(val_pred_test, y_val)
print('val_acc: %.3f' % val_acc1)
print('*' * 60)

y_test_proba = gbm.predict(x_test, num_iteration=gbm.best_iteration)
# y_pred = np.argmax(y_pred, axis=1)
print(y_test_proba.shape)
print(y_test_proba[:10])
y_test_pred = (y_test_proba>=0.5).astype(int)
print(y_test_pred[:10])

y_data_proba = gbm.predict(x_data, num_iteration=gbm.best_iteration)
# y_pred = np.argmax(y_pred, axis=1)
print(y_data_proba.shape)
print(y_data_proba[:10])
y_data_pred = (y_data_proba>=0.5).astype(int)
print(y_data_pred[:10])


# ## 4. Predict and Export titanic_pred.csv file

# In[29]:


random_num = str(int(random_num)).zfill(4)
print(random_num)

run_name_acc = run_name + '_' + str(int(val_acc*10000)).zfill(4) + '_' + str(int(val_acc1*10000)).zfill(4)
print(run_name_acc)


# In[30]:


def save_proba(y_data_proba, y_data, y_test_proba, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        print('Remove file: %s' % file_name)
    with h5py.File(file_name) as h:
        h.create_dataset('y_data_proba', data=y_data_proba)
        h.create_dataset('y_data', data=y_data)
        h.create_dataset('y_test_proba', data=y_test_proba)
    print('Save file: %s' % file_name)

def load_proba(file_name):
    with h5py.File(file_name, 'r') as h:
        y_data_proba = np.array(h['y_data_proba'])
        y_data = np.array(h['y_data'])
        y_test_proba = np.array(h['y_test_proba'])
    print('Load file: %s' % file_name)
    return y_data_proba, y_data, y_test_proba


# In[31]:


y_proba_file = os.path.join(model_temp_folder, 'titanic_proba_%s_%s.p' % (run_name_acc, random_num))
save_proba(y_data_proba, y_data, y_test_proba, y_proba_file)
y_data_proba, y_data, y_test_proba = load_proba(y_proba_file)

print(y_data_proba.shape)
print(y_data.shape)
print(y_test_proba.shape)


# In[32]:


passenger_id = data_test['PassengerId']
output = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': y_test_pred })

output_csv_file = os.path.join(output_temp_folder, '%s_%s.csv' % (run_name_acc, random_num))
output.to_csv(output_csv_file, index = False)
print(output_csv_file)
print('\n%s_%s' % (run_name_acc, random_num))


# In[33]:


print('val_acc: %.3f' % val_acc)
print('val_acc: %.3f' % val_acc1)


# In[34]:


print('Done!')

