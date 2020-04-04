# This is a script so we can do Live Share without the weird Jupyter thing happening

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac
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
        Box1List.append(float(entry[column]))
        Box2List.append(float(entry[column + 1]))
        Box3List.append(float(entry[column + 2]))
        Box4List.append(float(entry[column + 3]))

# Creating AllBoxes, which is a huge list of all SpectraMatrix data for easy tensor creation
AllBoxes = Box1List + Box2List + Box3List + Box4List

theTensor = np.array(AllBoxes).reshape((4, 200, 441))  # The tensor, in the proper shape, that we hope to do CPT on
# print(theTensor[0])
theCPT = parafac(theTensor, rank=4)
print("# of factors =",len(theCPT))    #3
print("# of matrix =",len(theCPT[0])) #4
print("# of row =",len(theCPT[1]))    #200
print("# of col =",len(theCPT[2]))    #441
#  theCPT[feature][row][col]
# print('feature 1:\n',theCPT[2][3][3])
# print(theCPT[0])
# y = [1,2,3,4]
plt.close()
plt.plot(theCPT[1],label = 'KUT')
plt.plot(theCPT[1], label = 'Tech-99')
plt.plot(theCPT[1], label = 'Cobalt-60')
plt.plot(theCPT[1], label = 'Cesium-137')

# plt.plot(theCPT[1], label=['cat', 'dog','fish','chicken'] )
plt.legend()

# feature2 = plt.plot(theCPT[2])
plt.show()

# Blue: KUT
# Red: Technetium-99m
# Orange: Cobalt-60
# Green: Cesium-137


# countTensor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]).reshape(2, 3, 3)
# randTensor = np.random.random((2, 2, 2)) # num of matrix, rows, cols 
# print('tensor = \n',countTensor)
# print()
# factors = parafac(countTensor, rank=3)
# print('factor = \n',factors)

