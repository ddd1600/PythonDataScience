import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense#, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import linear, relu, sigmoid

matplotlib widget
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from autils import *
from lab_utils_softmax import plt_softmax

np.set_printoptions(precision=2)

X,y = load_data()
