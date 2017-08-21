# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as numpy
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cwd = os.getcwd()


def dense_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


def create_record():
    num_classes = ['/data/0', '/data/1']
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(num_classes):
        class_path = cwd + name + "/"
        print(class_path)
        for img_name in os.listdir(class_path):
            if not img_name.startswith("."):
                img_path = class_path + img_name
                img = Image.open(img_path).convert("L")
                img = img.resize((28, 28))
                img_raw = img.tobytes() # 将图片转化为原生bytes
                example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
    writer.close()

def read_record():
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(["train.tfrecords"])
    _, serialized_example = reader.read(filename_queue)

    # 解析读取的样例
    features = tf.parse_single_example(
        serialized_example,
        features={
        'img_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
        })

    decoded_images = tf.decode_raw(features['img_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    labels = tf.cast(features['label'], tf.int32)
    images = tf.reshape(retyped_images, [784])
    return images, labels

# --------------------100一个batch 打包-------------------

create_record()
images, labels = read_record()

min_after_dequeue = 5000
BATCH_SIZE = 200
capacity = min_after_dequeue + 3 * BATCH_SIZE

image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                            batch_size=BATCH_SIZE,
                            capacity=capacity,
                            min_after_dequeue=min_after_dequeue)

# --------------------训练模型-------------------------
def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2

# 模型相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 2
LAYER1_NODE = 400
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000

x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
y_ = tf.placeholder(tf.int32, name='y-input')

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(x, weights1, biases1, weights2, biases2)

# 计算交叉熵及其平均值
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 正则化
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion


# 优化器
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

# 计算正确率
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 低度下降时本案例最佳参数：batch_size 200 layer1_node 400 learning_rate 0.0005

# 初始化会话，并开始训练过程。
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(500):
        xs, ys = sess.run([image_batch, label_batch])
        ys = dense_to_one_hot(ys, 2)
        sess.run(train_step, feed_dict={x: xs, y_: ys})
        if i % 10 == 0:
            print("%d step(s), loss --> %g " % (i, sess.run(loss, feed_dict={x: xs, y_: ys})))
            print("accuracy --> %g" % sess.run(accuracy, feed_dict={x: xs, y_: ys}))
            # print(sess.run(tf.arg_max(y, 1), feed_dict={x: xs, y_: ys}))
            # print(sess.run(tf.arg_max(ys, 1)))
