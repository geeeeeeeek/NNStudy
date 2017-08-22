# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("images/mock.png", "r",).read()

with tf.Session() as sess:

    # 解码图像
    img_data = tf.image.decode_png(image_raw_data)

    print img_data.eval()

    # 展示图像
    plt.imshow(img_data.eval())
    plt.show()

    # ---------------------调整尺寸-------------------------
    resized = tf.image.resize_images(img_data, [700, 700], method=0)
    # 转为uint才能展示
    resized = np.asarray(resized.eval(), dtype='uint8')
    plt.imshow(resized)
    plt.show()

    # ---------------------剪裁和填充-------------------------
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 100, 100)
    padding = tf.image.resize_image_with_crop_or_pad(img_data, 800, 800)

    plt.imshow(croped.eval())
    plt.show()

    plt.imshow(padding.eval())
    plt.show()

    # ----------------------截取50%-----------------------------
    center_croped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(center_croped.eval())
    plt.show()

    # ---------------------上下或左右翻转------------------------
    flip1 = tf.image.flip_up_down(img_data)
    flip2 = tf.image.flip_left_right(img_data)

    plt.imshow(flip1.eval())
    plt.show()

    plt.imshow(flip2.eval())
    plt.show()

    # -----------------------亮度调整----------------------------
    adjusted = tf.image.adjust_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # -----------------------添加标注框---------------------------
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)

    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.55]]])
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)

    distorted_image = tf.slice(img_data, begin, size)
    plt.imshow(distorted_image.eval())

    plt.show()


    # 转化为实数类型
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint16)
    # jpeg编码
    encoded_image = tf.image.encode_png(img_data)
