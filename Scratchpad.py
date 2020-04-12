# This is a scratchpad for doing quick operations.

# Most recently (2020.4.12) this was used to create some ideal
# spectra for the midterm presentation.

import pandas as pd
import ThingsWeDoALot as th
import matplotlib.pyplot as plt
import numpy as np
import time

thecenters = [880, 881, 882, 883]
cobaltROM = [1, 617, 705, 881]
SpectraMatrix = th.makeDataFrameWithTheXAxis(dataframefilepath='data files/SpectraMatrix.csv')


for i in cobaltROM:
    th.linePlotTheThing(SpectraMatrix,
                        i,
                        f'Sensor #{i}',
                        xaxislabel='Photon Energy (MeV)',
                        yaxislabel='Normalized Photon Frequency',
                        color='xkcd:cerulean blue',
                        islogscale=True,
                        figuredimensions=(4.5, 4.5))


