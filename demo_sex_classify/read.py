# -*- coding: utf-8 -*-
import csv
import os
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tfRecord的使用

cwd = os.getcwd()


def create_record():
    num_classes = ['/tf_test14/0', '/tf_test14/1', '/tf_test14/2']
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(num_classes):
        class_path = cwd + name + "/"
        print(class_path)
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()
