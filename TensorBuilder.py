# This is going to construct a NumPy tensor from the SpectraMatrix.csv file.
# By creating it from SpectraMatrix rather than simply stacking Box1, Box2, etc.,
# we should hopefully be able to prevent floating point errors.
# Oh hey, alternatively, maybe we can un-normalize the boxes and truncate them,
# then redo the normalization to chop the errors out. Let's try that in Scratchpad.py

import pandas as pd
import numpy as np
import tensorflow as tf
import ThingsWeDoALot as th

SpectraMatrix = th.makeDataFrameWithTheXAxis(dataframefilepath='data files/SpectraMatrix.csv')

datalist = [x for x in range(27)]
print(datalist)

exampleTensor = tf.constant(datalist, shape=(3, 3, 3))
tf.print(exampleTensor)
