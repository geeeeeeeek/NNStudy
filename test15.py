# -*- coding: utf-8 -*-
import numpy


def dense_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    # 行首偏移
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


labels = numpy.array([2, 1, 3, 4])

labels_one_hot = dense_to_one_hot(labels, 5)

print(labels_one_hot)
