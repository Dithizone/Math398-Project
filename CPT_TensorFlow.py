# This attempted CPT with TensorFlow, but was abandoned in favor of CPT_TEnsorly.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import ThingsWeDoALot as th

# Separating SpectraMatrix into each Box as a list
Box1List = []
Box2List = []
Box3List = []
Box4List = []

SpectraMatrixRows = open('data files/SpectraMatrix.csv', 'r').read().split(sep='\n')
for row in SpectraMatrixRows:
    entry = row.split(sep=',')
    for i in range(441):
        column = i * 4
        Box1List.append(entry[column])
        Box2List.append(entry[column + 1])
        Box3List.append(entry[column + 2])
        Box4List.append(entry[column + 3])

# Testing to ensure each Box contains the values we expect:
# print(f'Box 1 first: {Box1List[0]}, Box 1 last: {Box1List[-1]}')
# print(f'Box 2 first: {Box2List[0]}, Box 2 last: {Box2List[-1]}')
# print(f'Box 3 first: {Box3List[0]}, Box 3 last: {Box3List[-1]}')
# print(f'Box 4 first: {Box4List[0]}, Box 4 last: {Box4List[-1]}')

# Creating AllBoxes, which is a huge list of all SpectraMatrix data for easy tensor creation
AllBoxes = Box1List + Box2List + Box3List + Box4List

# Testing to ensure values make sense:
# print(f'AllBox first: {AllBoxes[0]}, AllBox last: {AllBoxes[-1]}')

TheTensor = tf.constant(AllBoxes, shape=(4, 200, 441))  # shape=(slices, rows, columns)
