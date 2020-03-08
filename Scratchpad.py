# This was a quick script to create Box 5, which is made of
# Box2 + Box3 + Box4 - 2*Box1, after unnormalizing (denormalizing? Strangifying?)

import pandas as pd
import matplotlib.pyplot as plt
import time

Box1 = pd.read_csv("data files/Box1.csv", header=None, index_col=None)
del Box1[441]

Box2 = pd.read_csv("data files/Box2.csv", header=None, index_col=None)
del Box2[441]

Box3 = pd.read_csv("data files/Box3.csv", header=None, index_col=None)
del Box3[441]

Box4 = pd.read_csv("data files/Box4.csv", header=None, index_col=None)
del Box4[441]

Box5 = pd.read_csv("data files/Box5.csv", header=None, index_col=None)

for i in range(21):
    Box5.plot(kind="line", y=(232+i), figsize=(8, 6), logy=True)
    plt.show()
    time.sleep(1.5)
