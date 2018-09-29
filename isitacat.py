import matplotlib as mpl
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard


# Marcus!!!! GLÖM EJ!!!!
mpl.use('TkAgg')


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
cat_ind = [index for index, value in enumerate(dic[b"labels"]) if value == name_index]


def getdata(start, end):
    x = []
    for i in range(start, end):
        img = np.array(dic[b"data"][i]).reshape(3, 32, 32)
        img = np.dstack((img[0], img[1], img[2]))
        x.append(img)

    x = np.array(x)
    y = np.array(dic[b"labels"])[start:end]
    y[y != name_index] = 0
    y[y == name_index] = 1

    return x / 255, np_utils.to_categorical(y)


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


X_train, Y_train = getdata(0, 8000)
X_test, Y_test = getdata(8001, 10000)

num_classes = Y_test.shape[1]

model = base_model()

tb = TensorBoard(log_dir='./logs/initial_setting')
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=40, batch_size=128, callbacks=[tb])
model.save('my_model_cat_gpu.h5')

