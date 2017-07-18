# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# ------------------------不好的写法：将学习率写死的情况-------------------------------
# TRAINING_STEPS = 2000
# LEARNING_RATE = 0.01
# x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
# y = tf.square(x)
#
# train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(TRAINING_STEPS):
#         sess.run(train_op)
#         if i % 100 == 0:
#             x_value = sess.run(x)
#             print "After %s iteration(s): x=%s is %f." % (i + 1, i + 1, x_value)


# -------------------------好的写法：指数衰减学习率---------------------------------
TRAINING_STEPS = 100
global_step = tf.Variable(0)
# 初始0.1 衰减率0.96
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)

x = tf.Variable(tf.constant(10, dtype=tf.float32), name="x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 1 == 0:
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            x_value = sess.run(x)
            print "After %s iteration(s): x%s is %f, learning rate is %f."% (i+1, i+1, x_value, LEARNING_RATE_value)
