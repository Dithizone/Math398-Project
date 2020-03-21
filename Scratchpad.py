# TensorFlow can use CUDA 10.1 to accelerate tensor stuff using
# the GPU. Currently it lags a bit when the code runs because
# the library is searching for CUDA stuff and can't find it,
# so switches to CPU instead. If we start doing crazy stuff,
# it might be interesting to employ the GPU to do calculations
# really really fast.

import pandas as pd
import ThingsWeDoALot as th
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf

SpectraMatrix = th.makeDataFrameWithTheXAxis(dataframefilepath='data files/SpectraMatrix.csv')

thetensor = tf.constant([
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
    [[10, 11, 12],
     [13, 14, 15],
     [16, 17, 18]],
    [[19, 20, 21],
     [22, 23, 24],
     [25, 26, 27]]])

tf.print(thetensor)
tf.print(thetensor.shape)

