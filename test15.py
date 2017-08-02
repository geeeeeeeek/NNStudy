# -*- coding: utf-8 -*-
import numpy

num_labels = 5
num_classes = 10

# 行首偏移
index_offset = numpy.arange(num_labels) * num_classes

print(index_offset)

# labels
labels = numpy.array([2, 3, 4, 2, 9])

print(labels)

labels_one_hot = numpy.zeros((num_labels, num_classes))

labels_one_hot.flat[index_offset + labels.ravel()] = 1

print(labels_one_hot)
