# This was a quick script to create Box 5, which is made of
# Box2 + Box3 + Box4 - 2*Box1, after unnormalizing (denormalizing? Strangifying?)

import pandas as pd
import ThingsWeDoALot as th
import matplotlib.pyplot as plt
import numpy as np
import time

Box1withoutXAxis = pd.read_csv('data files/Box1.csv', header=None, index_col=None)
Box1 = th.attachXAxis(dataframe=Box1withoutXAxis)
print(Box1)
th.linePlotTheThing(dataframetoplot=Box1, columntoplot=221, xaxislabel='MeV', title='Spectrum at sensor 221', islogscale=True)