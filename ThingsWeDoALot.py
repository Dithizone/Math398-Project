# It'd be so awesome to write functions here which we can call
# in other files or notebooks! Let's see if it can be done.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def makeDataFrameWithTheXAxis(dataframefilepath, thexaxisfilepath='data files/TheXAxis.csv'):
    needsXAxis = pd.read_csv(filepath_or_buffer=dataframefilepath, header=None, index_col=None)
    theXAxis = pd.read_csv(filepath_or_buffer=thexaxisfilepath, index_col=None)
    hasXAxisNow = pd.concat([needsXAxis, theXAxis], axis=1)
    hasXAxisAsIndex = hasXAxisNow.set_index('theXAxis')
    return hasXAxisAsIndex


def attachXAxis(dataframe, thexaxisfilepath='data files/TheXAxis.csv'):
    theXAxis = pd.read_csv(filepath_or_buffer=thexaxisfilepath, index_col=None)
    hasXAxisNow = pd.concat((dataframe, theXAxis), axis=1)
    hasXAxisAsIndex = hasXAxisNow.set_index('theXAxis')
    return hasXAxisAsIndex


def linePlotTheThing(dataframetoplot, columntoplot, title=None, titlefontsize=15, xaxislabel=None, yaxislabel=None, islogscale=False, figuredimensions=(8, 6), filepathtosavepng=None):
    dataframetoplot.plot(kind='line', y=columntoplot, figsize=figuredimensions, logy=islogscale)
    plt.xlabel(xlabel=xaxislabel)
    plt.ylabel(ylabel=yaxislabel)
    plt.title(label=title, fontsize=titlefontsize)
    if filepathtosavepng is not None:
        plt.savefig(fname=filepathtosavepng, bbox_inches='tight', orientation="landscape", pad_inches=0.2, dpi=600)
    return plt.show()

