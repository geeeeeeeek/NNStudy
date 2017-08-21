# -*- coding: utf-8 -*-
import numpy as numpy


def dense_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
