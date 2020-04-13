# This is a scratchpad for doing quick operations.

# Most recently (2020.4.12) this was used to create some ideal
# spectra for the midterm presentation.

import pandas as pd
import ThingsWeDoALot as th
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

thecenters = [880, 881, 882, 883]
cobaltROM = [1, 617, 705, 881]
PCsWeWant = [0, 1, 2]
SpectraMatrix = th.makeDataFrameWithTheXAxis(dataframefilepath='data files/SpectraMatrix.csv')

x = StandardScaler().fit_transform(SpectraMatrix.loc[:, :].values)
pca = PCA(n_components=10).fit(x)
principalComponents = pca.fit_transform(x)
principalDataframe = pd.DataFrame(data=principalComponents)
princicalDataframeWXAxis = th.attachXAxis(principalDataframe)

plt.figure(figsize=(10, 8))
plt.xlabel('Photon Energy (MeV)')
plt.ylabel('Value in Principal Component')
plt.title('SpectraMatrix Principal Components', fontsize=15)

for j in PCsWeWant:
    plt.plot(princicalDataframeWXAxis.iloc[:, j], label=f'Principal Component {j + 1}')
plt.legend()
th.saveThisGraph('images/midtermPresentation/PC123.png')
plt.show()

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

