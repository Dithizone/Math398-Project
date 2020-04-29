# This used to be a scratchpad but we used it for something
# cool that's worth saving as a dedicated script.

# Most recently (2020.4.28) this was used to create some ideal
# spectra for the midterm presentation.

import pandas as pd
import ThingsWeDoALot as th
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

thecenters = [880, 881, 882, 883]
cobaltROM = [1, 617, 705, 881]
PCsWeWant = [2, 3, 4]
SpectraMatrix = th.makeDataFrameWithTheXAxis(dataframefilepath='data files/SpectraMatrix.csv')

# Doing PCA on our data
x = StandardScaler().fit_transform(SpectraMatrix.loc[:, :].values)
pca = PCA(n_components=10).fit(x)
principalComponents = pca.fit_transform(x)
principalDataframe = pd.DataFrame(data=principalComponents)
princicalDataframeWXAxis = th.attachXAxis(principalDataframe)

# ------- The PCA Subplots --------
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Value in Principal Component', fontsize=13)
plt.title('Principal Component 1 of Spectra Data Set', fontsize=15)
plt.plot(princicalDataframeWXAxis.iloc[:, 0], label=f'Principal Component 1')

plt.subplot(1, 2, 2)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.title('Principal Components 3, 4, and 5', fontsize=15)

for j in PCsWeWant:
    plt.plot(princicalDataframeWXAxis.iloc[:, j], label=f'Principal Component {j + 1}')
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
plt.legend(fontsize=16)

plt.subplot(2, 2, 2)
# plt.xlabel('Photon Energy (MeV)', fontsize=13)
# plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title(' ', fontsize=40)
plt.plot(SpectraMatrix.iloc[:, thecenters[1]], label=f'Cobalt-60')
plt.yscale('log')
plt.legend()

plt.subplot(2, 2, 3)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
plt.ylabel('Normalized Frequency of Detected Photons', fontsize=13)
# plt.title('W Matrix Column 3', fontsize=15)
plt.plot(SpectraMatrix.iloc[:, thecenters[2]], label=f'Cesium-137', color='orange')
plt.yscale('log')
plt.legend()

plt.subplot(2, 2, 4)
plt.xlabel('Photon Energy (MeV)', fontsize=13)
# plt.ylabel('Value After Decomposition', fontsize=13)
# plt.title('W Matrix Column 4', fontsize=15)
plt.plot(SpectraMatrix.iloc[:, thecenters[3]], label=f'Technetium-99m', color='green')
plt.yscale('log')
plt.legend()

# th.saveThisGraph('images/subplots/theSpectra.png')
plt.show()
plt.close()