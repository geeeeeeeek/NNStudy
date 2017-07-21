# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# ----------保存模型的方法---------------

MODEL_PATH = "models/test10/model.ckpt";

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init_op)
    # 保存模型
    saver.save(sess, MODEL_PATH)


# 使用模型
with tf.Session() as sess:
    saver.restore(sess, MODEL_PATH)
    print sess.run(result)
