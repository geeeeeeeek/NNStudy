# -*- coding: utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState

batch_size = 10

dataset_size = 100

# 生成模拟集
def mock_dataset():
    rdm = RandomState(1)
    X = rdm.rand(dataset_size, 2)
    Y = [[int(x1 + x2 < 1)] for(x1, x2) in X]
    return X, Y


def fetch_batch(index, src_x, src_y):
    start = (index * batch_size) % dataset_size
    end = min(start + batch_size, dataset_size)
    return src_x[start:end], src_y[start:end]


def train():

    X, Y = mock_dataset()

    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1)))

    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            currentX, currentY = fetch_batch(i, X, Y)
            sess.run(train_step, feed_dict={x: currentX, y_: currentY})
            if i % 100 == 0:
                total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
                print("%d steps is %g" % (i, total_cross_entropy))
                print(sess.run(y, feed_dict={x: currentX, y_: currentY}))

if __name__ == '__main__':
    train()
