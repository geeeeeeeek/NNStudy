# -*- coding: utf-8 -*-
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def compute_cross_entropy_with_softmax():
    y = tf.constant([[1.0, 2.0], [0.2, 1.0], [0.0, 1.0]])
    y_ = tf.constant([[0, 1], [1, 0], [1, 0]])
    # logits labels shape must be-->  [batch_size, num_classes]
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    with tf.Session() as sess:
        print(sess.run(cross_entropy))
        print(sess.run(cross_entropy_mean))


def compute_cross_entropy_with_sparse():
    y = tf.constant([[0.1, 0.9], [0.8, 0.1], [0.8, 0.1]])
    y_ = tf.constant([1, 0, 0])
    # y_是y中num_classes的索引
    # logits labels shape must be--> [batch_size, num_classes] and [batch_size]
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    with tf.Session() as sess:
        print(sess.run(cross_entropy))
        print(sess.run(cross_entropy_mean))

if __name__ == '__main__':
    compute_cross_entropy_with_softmax()
    compute_cross_entropy_with_sparse()
