# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image
import utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cwd = os.getcwd()


def create_record(data_root, record_name):
    writer = tf.python_io.TFRecordWriter(record_name)
    for index, name in enumerate(data_root):
        class_path = cwd + name + "/"
        print(class_path)
        for img_name in os.listdir(class_path):
            if not img_name.startswith("."):
                img_path = class_path + img_name
                img = Image.open(img_path).convert("L")
                img = img.resize((28, 28))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
    writer.close()


def read_record(record_name):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([record_name])
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


def get_batch(images, labels, batch_size):
    image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                      batch_size=batch_size,
                                                      capacity=8000,
                                                      min_after_dequeue=5000)
    return image_batch, label_batch


def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


# 低度下降时本案例最佳参数：batch_size 200 layer1_node 400 learning_rate 0.0005
# AdamOptimizer 最佳参数：batch_size 200 layer1_node 50 learning_rate 0.001

# 模型相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 2
LAYER1_NODE = 50
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
BATCH_SIZE = 200
LEARNING_RATE = 0.001

data_train_root = ["/data/train/0", "/data/train/1"]
data_validate_root = ["/data/validate/0", "/data/validate/1"]
record_train_name = "train.tfrecords"
record_validate_name = "validate.tfrecords"


def train():

    # 读取record
    train_images, train_labels = read_record(record_train_name)
    validate_images, validate_labels = read_record(record_validate_name)

    # 获取batch
    train_image_batch, train_label_batch = get_batch(train_images, train_labels, BATCH_SIZE)
    validata_image_batch, validate_label_batch = get_batch(validate_images, validate_labels, BATCH_SIZE)

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
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # 计算正确率
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(500):
            train_x, train_y = sess.run([train_image_batch, train_label_batch])
            train_y = utils.dense_to_one_hot(train_y, 2)
            sess.run(train_step, feed_dict={x: train_x, y_: train_y})
            # if i % 10 == 0:
            # print(sess.run(tf.arg_max(y, 1), feed_dict={x: xs, y_: ys}))
            # print(sess.run(tf.arg_max(ys, 1)))
            if i % 10 == 0:
                print("\n%d steps" % i)
                print("loss --> %g " % sess.run(loss, feed_dict={x: train_x, y_: train_y}))
                validate_x, validate_y = sess.run([validata_image_batch, validate_label_batch])
                validate_y = utils.dense_to_one_hot(validate_y, 2)
                print("accuracy --> %g" % sess.run(accuracy, feed_dict={x: validate_x, y_: validate_y}))


if __name__ == '__main__':

    # 训练集
    create_record(data_train_root, record_train_name)

    # 验证集
    create_record(data_validate_root, record_validate_name)

    train()
