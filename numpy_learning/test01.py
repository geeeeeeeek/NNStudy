# -*- coding: utf-8 -*-
import numpy as np

# reshape的使用

aa = np.array([10, 11, 13, 12, 21, 23, 22, 22, 33, 34, 31, 44, 43, 41, 49, 55, 59, 52])

bb = aa.reshape((3, 3, 2))

print bb
