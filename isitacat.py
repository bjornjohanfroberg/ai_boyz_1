import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dic = unpickle('cifar-10-batches-py/data_batch_1')

print(dic)

imgR = dic[b"data"][100][:1024]
imgR = np.reshape(img, (32, 32))
imgplot = plt.imshow(img)
plt.show()


print(img)
