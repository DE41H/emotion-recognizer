import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

PATH = './data'
EPOCHS = 20
LEARNING_RATE = 0.001

def load():
    x = np.load(os.path.join(PATH, 'features.npy'))
    y = np.load(os.path.join(PATH, 'labels.npy'))
    g = np.load(os.path.join(PATH, 'genders.npy'))
    a = np.load(os.path.join(PATH, 'actors.npy'))
    return x, y, g, a

def split(x, y, a, g):
    x_train, x_temp, y_train, y_temp, a_train, a_temp, g_train, g_temp = train_test_split(x, y, a, g, test_size=0.2, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test, a_val, a_test, g_val, g_test = train_test_split(x_temp, y_temp, a_temp, g_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return (x_train, x_test, x_val), (y_train, y_test, y_val), (a_train, a_test, a_val), (g_train, g_test, g_val)
