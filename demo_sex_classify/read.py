# -*- coding: utf-8 -*-
import csv
import os
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tfRecord的使用

cwd = os.getcwd()


def create_record(filename):
    num_classes = ['/data/0', '/data/1']
    writer = tf.python_io.TFRecordWriter(filename)
    for index, name in enumerate(num_classes):
        class_path = cwd + name + "/"
        print(class_path)
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path).convert("L")
            img = img.resize((28, 28))
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def read_record(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
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

create_record("test.tfrecords")
img, label = read_record("test.tfrecords")

img_batch, label_batch = tf.train.shuffle_batch([img, label],
                        batch_size=30, capacity=2000,
                        min_after_dequeue=1000)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l = sess.run([img_batch, label_batch])
        print(val.shape, l)