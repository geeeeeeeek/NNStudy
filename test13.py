# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("images/", "r",).read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    print img_data.eval()
