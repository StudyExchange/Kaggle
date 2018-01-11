# -*- coding: utf-8 -*-

import time
import sys
import os

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19, preprocess_input

from sklearn.model_selection import train_test_split, KFold