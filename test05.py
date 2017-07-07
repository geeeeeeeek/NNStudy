import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

w1 = tf.Variable(tf.random_normal([2,3], stddev=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1))

x = tf.placeholder(tf.float32, shape=(1,2), name = "input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7,0.9]]}))

