import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib as mpl
mpl.use('TkAgg')
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
first = dic[b"labels"].index(name_index)



#plt.imshow(img)
#plt.show()


cat_ind = [index for index, value in enumerate(dic[b"labels"]) if value == name_index]

print("There is %d cats!!!!!! :D" % len(cat_ind))

#x = tf.placeholder(tf.float32, [None, len(dic[b"data"][first])])
x = tf.placeholder(tf.float32, [None, 32, 32, 3])

x_image = tf.reshape(x, [-1, 32, 32, 3])

# Layer 1
W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 42], stddev=0.1))
b1 = tf.Variable(tf.constant(0.12, shape=[42]))
conv1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
y1 = tf.nn.relu(conv1 + b1)
pool1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Layer 2
W2 = tf.Variable(tf.truncated_normal([5, 5, 42, 88], stddev=0.1))
b2 = tf.Variable(tf.constant(0.12, shape=[88]))
conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME')
y2 = tf.nn.relu(conv2 + b2)
pool2 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1] , padding="SAME")

# Densely Connected Layer
pool2_platt = tf.reshape(pool2, [-1, 8*8*88])
W3 = tf.Variable(tf.truncated_normal([8*8*88, 1025], stddev=0.1))
b3 = tf.Variable(tf.constant(0.12, shape=[1025]))
y_dcl = tf.nn.relu(tf.matmul(pool2_platt, W3) + b3)

# Read out
W4 = tf.Variable(tf.truncated_normal([1025, 1], stddev=0.1))
b4 = tf.Variable(tf.constant(0.12, shape=[1]))
y_out = tf.matmul(y_dcl, W4) + b4

y_ = tf.placeholder(tf.float32, [None, 1])

# Train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.round(y_out), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

index = 5000
batch_x_test = []
batch_y_test = []
for _ in range(700):
    # Create image
    img_big = dic[b"data"][index]
    img = []
    for i in range(1024):
        img.append([img_big[i], img_big[i + 1024], img_big[i + 1024 * 2]])
    img = np.reshape(img, (32, 32, 3))
    batch_x_test.append(img)
    batch_y_test.append([dic[b"labels"][index]])
    index = index + 1




index = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(100):
        batch_x = []
        batch_y = []
        for _ in range(70):
            # Create image
            img_big = dic[b"data"][index]
            img = []
            for i in range(1024):
                img.append([img_big[i], img_big[i + 1024], img_big[i + 1024 * 2]])
            img = np.reshape(img, (32, 32, 3))
            batch_x.append(img)
            batch_y.append([int(dic[b"labels"][index] == 3)])
            index = index + 1

        print("step: %d" % j)

        print(sess.run(y_out, feed_dict={x: batch_x, y_: batch_y}))
        if j % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_x_test, y_: batch_y_test})
            print('step %d, training accuracy %g' % (j, train_accuracy))
        train_step.run(feed_dict={x: batch_x, y_: batch_y})

# print('test accuracy %g' % accuracy.eval(feed_dict={
    # x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



