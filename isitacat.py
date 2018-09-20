import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# One-hot encoding
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping
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


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Find cat
names = unpickle('cifar-10-batches-py/batches.meta')
name_index = names[b"label_names"].index(b"cat")

# Get data
dic = unpickle('cifar-10-batches-py/data_batch_1')

#IMG = np.array(dic[b"data"]).reshape(3, 32, 32, 10000)
IMG = []
for i in range(0, 10000):
    img = np.array(dic[b"data"][i]).reshape(3, 32, 32)
    img = np.dstack((img[0], img[1], img[2]))
    IMG.append(img)

IMG = np.array(IMG)
print(IMG.shape)
X_train = IMG
X_test = X_train
X_train = X_train / 255
X_test = X_test / 255

'''
# Find index of first cat
first = dic[b"labels"].index(name_index)
img = np.array(dic[b"data"][first]).reshape(3, 32, 32)
img = np.dstack((img[0], img[1], img[2]))
plt.imshow(img)
plt.show()
print(img.shape)
'''

cat_ind = [index for index, value in enumerate(dic[b"labels"]) if value == name_index]

print("There is %d cats!!!!!! :D" % len(cat_ind))
y_train = np.array(dic[b"labels"])
y_train[y_train != name_index] = 0
y_train[y_train == name_index] = 1
y_test = y_train

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(num_classes)

def base_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = base_model()

# Fit the model
tb = TensorBoard(log_dir='./logs/initial_setting')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=128, callbacks=[tb])
model.save('my_model_cat.h5')
