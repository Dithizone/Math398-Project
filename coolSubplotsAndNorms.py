# This used to be a scratchpad but we used it for something
# cool that's worth saving as a dedicated script.

# Most recently (2020.05.05) this was used to determine norms
# and perform reconstructions.

import pandas as pd
import ThingsWeDoALot as th
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from numpy.linalg import norm
import tensorly as tl
from tensorly.decomposition import non_negative_parafac

thecenters = [880, 881, 882, 883]
cobaltROM = [1, 617, 705, 881]
PCsWeWant = [2, 3, 4]
SpectraMatrix = th.makeDataFrameWithTheXAxis(dataframefilepath='data files/SpectraMatrix.csv')

# Doing PCA on our data
x = StandardScaler().fit_transform(SpectraMatrix.loc[:, :].values)
pca = PCA(n_components=10).fit(x)
principalComponents = pca.fit_transform(x)
principalDataframe = pd.DataFrame(data=principalComponents)
principalDataframeWXAxis = th.attachXAxis(principalDataframe)

# ------- The PCA Subplots --------
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Value in Principal Component', fontsize=13)
plt.title('Principal Component 1 of Spectra Data Set', fontsize=15)
plt.plot(principalDataframeWXAxis.iloc[:, 0], label=f'Principal Component 1')

plt.subplot(1, 2, 2)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.title('Principal Components 3, 4, and 5', fontsize=15)

for j in PCsWeWant:
    plt.plot(principalDataframeWXAxis.iloc[:, j], label=f'Principal Component {j + 1}')
plt.legend()
# th.saveThisGraph('images/subplots/PC1235.png')
plt.show()

plt.close()
# ------- The Graphs to illustrate a ROM -------
# for i in cobaltROM:
#     th.linePlotTheThing(SpectraMatrix,
#                         i,
#                         f'Sensor #{i}',
#                         xaxislabel='Photon Energy (MeV)',
#                         yaxislabel='Normalized Photon Frequency',
#                         color='xkcd:cerulean blue',
#                         islogscale=True,
#                         figuredimensions=(4.5, 4.5))
#

# ------- The NMF W Matrix Graphs

SpectraMatrixByBoxes = th.makeDataFrameWithTheXAxis(dataframefilepath='data files/SpectraMatrixByBoxes.csv')
modelSpectraMatrixByBoxes = NMF(n_components=4, init='random', solver='cd', beta_loss=2, tol=1e-10)
WSpectraMatrixByBoxes = modelSpectraMatrixByBoxes.fit_transform(SpectraMatrixByBoxes)
HSpectraMatrixByBoxes = modelSpectraMatrixByBoxes.components_

WSpectraMatrixByBoxesDataFrame = pd.DataFrame(data=WSpectraMatrixByBoxes)
WSpectraMatrixByBoxesDataFrame = th.attachXAxis(dataframe=WSpectraMatrixByBoxesDataFrame)

HSpectraMatrixByBoxesDataFrame = pd.DataFrame(data=HSpectraMatrixByBoxes)
# HSpectraMatrixByBoxesDataFrame = th.attachXAxis(dataframe=HSpectraMatrixByBoxesDataFrame)

plt.figure(figsize=(15, 9))
plt.suptitle('W Matrix of Spectra Data Set', fontsize=22)

plt.subplot(2, 2, 1)
# plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title(' ', fontsize=40)
plt.plot(WSpectraMatrixByBoxesDataFrame.iloc[:, 0], label=f'Column 1', color='xkcd:cerulean blue')
plt.legend()

plt.subplot(2, 2, 2)
# plt.xlabel('Photon Energy (MeV)', fontsize=13)
# plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title(' ', fontsize=40)
plt.plot(WSpectraMatrixByBoxesDataFrame.iloc[:, 1], label=f'Column 2', color='green')
plt.legend()

plt.subplot(2, 2, 3)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title('W Matrix Column 3', fontsize=15)
plt.plot(WSpectraMatrixByBoxesDataFrame.iloc[:, 2], label=f'Column 3')
plt.legend()

plt.subplot(2, 2, 4)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
# plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title('W Matrix Column 4', fontsize=15)
plt.plot(WSpectraMatrixByBoxesDataFrame.iloc[:, 3], label=f'Column 4', color='orange')
plt.legend()

# th.saveThisGraph('images/subplots/WMatrix.png')
plt.show()
plt.close()

# ------- The NMF H Matrix Graphs -------

modelSpectraMatrix = NMF(n_components=4, init='random', solver='cd', beta_loss=2, tol=1e-10)
WSpectraMatrix = modelSpectraMatrix.fit_transform(SpectraMatrix)
HSpectraMatrix = modelSpectraMatrix.components_
WSpectraMatrixDataFrame = pd.DataFrame(data=WSpectraMatrix)
HSpectraMatrixDataFrame = pd.DataFrame(data=HSpectraMatrix)

plt.figure(figsize=(15, 9))
plt.suptitle('H Matrix of Spectra Data Set', fontsize=22)

plt.subplot(2, 2, 1)
# plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title(' ', fontsize=40)
plt.plot(HSpectraMatrixDataFrame.iloc[0, :], label=f'Row 1', color='xkcd:cerulean blue')
plt.legend()

plt.subplot(2, 2, 2)
# plt.xlabel('Photon Energy (MeV)', fontsize=13)
# plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title(' ', fontsize=40)
plt.plot(HSpectraMatrixDataFrame.iloc[1, :], label=f'Row 2', color='xkcd:cerulean blue')
plt.legend()

plt.subplot(2, 2, 3)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title('W Matrix Column 3', fontsize=15)
plt.plot(HSpectraMatrixDataFrame.iloc[2, :], label=f'Row 3', color='xkcd:cerulean blue')
plt.legend()

plt.subplot(2, 2, 4)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
# plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title('W Matrix Column 4', fontsize=15)
plt.plot(HSpectraMatrixDataFrame.iloc[3, :], label=f'Row 4', color='xkcd:cerulean blue')
plt.legend()

# th.saveThisGraph('images/subplots/HMatrix.png')
plt.show()
plt.close()

# ------- Pretty graph of the centers -------

plt.figure(figsize=(12, 9))
plt.suptitle('Photon Energy Spectra', fontsize=22)

plt.subplot(2, 2, 1)
# plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Normalized Frequency of Detected Photons', fontsize=13)
# plt.title(' ', fontsize=40)
plt.plot(SpectraMatrix.iloc[:, thecenters[0]], label=f'Background', color='xkcd:cerulean blue')
plt.yscale('log')
plt.legend(fontsize=14)

plt.subplot(2, 2, 2)
# plt.xlabel('Photon Energy (MeV)', fontsize=13)
# plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title(' ', fontsize=40)
plt.plot(SpectraMatrix.iloc[:, thecenters[1]], label=f'Cobalt-60')
plt.yscale('log')
plt.legend(fontsize=14)

plt.subplot(2, 2, 3)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Normalized Frequency of Detected Photons', fontsize=13)
# plt.title('W Matrix Column 3', fontsize=15)
plt.plot(SpectraMatrix.iloc[:, thecenters[2]], label=f'Cesium-137', color='orange')
plt.yscale('log')
plt.legend(fontsize=14)

plt.subplot(2, 2, 4)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
# plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title('W Matrix Column 4', fontsize=15)
plt.plot(SpectraMatrix.iloc[:, thecenters[3]], label=f'Technetium-99m', color='green')
plt.yscale('log')
plt.legend(fontsize=14)

# th.saveThisGraph('images/subplots/theSpectra.png')
plt.show()
plt.close()

# ------- Doing CPT -------
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

theCPT = non_negative_parafac(theTensor, rank=4)  # The CPT decomposition

# ------- The Norms of the original dataset -------
frobeniusNormOfOriginalData = norm(SpectraMatrix, ord='fro')
L1NormOfOriginalData = norm(SpectraMatrix, ord=1)
L2NormOfOriginalData = norm(SpectraMatrix, ord=2)

frobeniusNormOftheTensor = norm(theTensor, ord='fro', axis=(1, 2))
L1NormOftheTensor = norm(theTensor, ord=1, axis=(1, 2))
L2NormOftheTensor = norm(theTensor, ord=2, axis=(1, 2))

# -------  X - W*H -------
numpyW = WSpectraMatrixDataFrame.to_numpy()
numpyH = HSpectraMatrixDataFrame.to_numpy()
numpySpectra = SpectraMatrix.to_numpy()
reconstructedNMF = np.matmul(numpyW, numpyH)
frobeniusNormOfreconstructedNMF = norm(reconstructedNMF, ord='fro')
L1NormOfreconstructedNMF = norm(reconstructedNMF, ord=1)
L2NormOfreconstructedNMF = norm(reconstructedNMF, ord=2)

# ------- CPT reconstruction
reconstructedCPT = tl.kruskal_to_tensor(theCPT)
frobeniusNormOfreconstructedCPT = norm(reconstructedCPT, ord='fro', axis=(1, 2))
L1NormOfreconstructedCPT = norm(reconstructedCPT, ord=1, axis=(1, 2))
L2NormOfreconstructedCPT = norm(reconstructedCPT, ord=2, axis=(1, 2))

# ------- All the norms -------
print(f'Frobenius Norms:')
print(f'Original: {frobeniusNormOfOriginalData}\n'
      f'NMF: {frobeniusNormOfreconstructedNMF}\n'
      f'Original Tensor: {frobeniusNormOftheTensor}\n'
      f'CPT: {frobeniusNormOfreconstructedCPT}\n')
print(f'L1 Norms:')
print(f'Original: {L1NormOfOriginalData}\n'
      f'NMF: {L1NormOfreconstructedNMF}\n'
      f'Original Tensor: {frobeniusNormOftheTensor}\n'
      f'CPT: {L1NormOfreconstructedCPT}\n')
print(f'L2 Norms:')
print(f'Original: {L2NormOfOriginalData}\n'
      f'NMF: {L2NormOfreconstructedNMF}\n'
      f'Original Tensor: {frobeniusNormOftheTensor}\n'
      f'CPT: {L2NormOfreconstructedCPT}\n')
