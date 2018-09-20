# The package for importing the dataset (already provided by Keras)
from keras.datasets import mnist

# Packages for defining the architecture of our model
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D

# One-hot encoding
from keras.utils import np_utils

# Callbacks for training
from keras.callbacks import TensorBoard, EarlyStopping

# Ploting
import matplotlib.pyplot as plt

# Ndarray computations
import numpy as np

# Confusion matrix for assessment step
#from sklearn.metrics import confusion_matrix

(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.imshow(X_train[10], cmap='gray')

