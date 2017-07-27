# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("images/mock.png", "r",).read()

with tf.Session() as sess:

    # 解码图像
    img_data = tf.image.decode_png(image_raw_data)

    print img_data.eval()

    # 展示图像
    plt.imshow(img_data.eval())
    plt.show()

    # 转化为实数类型
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint16)

    # jpeg编码
    encoded_image = tf.image.encode_png(img_data)
