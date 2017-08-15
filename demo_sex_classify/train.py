# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


cwd = os.getcwd()


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
                img = img.resize((60, 60))
                img_raw = img.tobytes()  # 将图片转化为原生bytes
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
    images = tf.reshape(retyped_images, [3600])
    return images, labels


# --------------------100一个batch 打包-------------------

#create_record()
images, labels = read_record()

min_after_dequeue = 1000
batch_size = 10
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                  batch_size=batch_size,
                                                  capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)


# --------------------训练模型-------------------------
def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


# 模型相关的参数
INPUT_NODE = 3600
OUTPUT_NODE = 2
LAYER1_NODE = 5
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)

# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化会话，并开始训练过程。
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i % 10 == 0:
            print("After %d training step(s), loss is %g " % (i, sess.run(loss)))

        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)
