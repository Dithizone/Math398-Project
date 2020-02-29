import pandas as pd
import matplotlib.pyplot as plt
import time

Box1 = pd.read_csv("data files/Box1.csv", header=None, index_col=None)
del Box1[441]
# Box1.info()

Box2 = pd.read_csv("data files/Box2.csv", header=None, index_col=None)
del Box2[441]
# Box2.info()

Box3 = pd.read_csv("data files/Box3.csv", header=None, index_col=None)
del Box3[441]
# Box3.info()

Box4 = pd.read_csv("data files/Box4.csv", header=None, index_col=None)
del Box4[441]
# Box4.info()

SpectraMatrixByBoxes1 = pd.DataFrame(data=Box1)
print(SpectraMatrixByBoxes1.head())
SpectraMatrixByBoxes12 = pd.concat([SpectraMatrixByBoxes1, Box2], axis=1, ignore_index=True)
SpectraMatrixByBoxes123 = pd.concat([SpectraMatrixByBoxes12, Box3], axis=1, ignore_index=True)
SpectraMatrixByBoxes = pd.concat([SpectraMatrixByBoxes123, Box4], axis=1, ignore_index=True)

print(SpectraMatrixByBoxes.head())

# diditwork = SpectraMatrixByBoxes.transpose()
# for i in range(20):
#     diditwork.plot(kind="line", y=(i+167), figsize=(8, 6))
#     plt.show()
#     time.sleep(1)

# SpectraMatrixByBoxes.to_csv(r'data files/SpectraMatrixByBoxes.csv', index=False, header=False)

testdataframe = pd.read_csv('data files/SpectraMatrixByBoxes.csv', header=None, index_col=None)
print(testdataframe.head())  # Yay it works
