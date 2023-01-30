import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model

