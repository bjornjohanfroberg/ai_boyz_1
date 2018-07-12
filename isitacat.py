import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


names = unpickle('cifar-10-batches-py/batches.meta')
print(names)

dic = unpickle('cifar-10-batches-py/data_batch_1')

img_big = dic[b"data"][123]
img = []
for i in range(1024):
    img.append([img_big[i], img_big[i+1024], img_big[i+1024*2]])


img = np.reshape(img, (32, 32, 3))


imgplot = plt.imshow(img)
plt.show()


print(img)
