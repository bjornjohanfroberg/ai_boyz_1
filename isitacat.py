import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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

# Find index of first cat
lab = dic[b"labels"]
first = lab.index(name_index)

# Create image
img_big = dic[b"data"][first]
img = []
for i in range(1024):
    img.append([img_big[i], img_big[i+1024], img_big[i+1024*2]])

img = np.reshape(img, (32, 32, 3))
plt.imshow(img)
plt.show()
