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
from keras.models import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_test[2])

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train / 255
X_test = X_test / 255

model = load_model('my_model.h5')

print(model.predict(X_test[2:3]))

plt.show()
