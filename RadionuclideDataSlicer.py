# Dicing up the SpectraMatrix data into its respective boxes

import pandas as pd

box1 = open("Box1.csv", "w")  # This will be 0, 4, 8, 12... 1760
box2 = open("Box2.csv", "w")  # 1, 5, 9, 13... 1761
box3 = open("Box3.csv", "w")  # 2, 6, 10, 14... 1762
box4 = open("Box4.csv", "w")  # 3, 7, 11, 15... 1763

SpectraMatrix = pd.read_csv("SpectraMatrix.csv", header=None, index_col=None)

file1 = open("Box1.csv", mode="w", newline="")
file2 = open("Box2.csv", mode="w", newline="")
file3 = open("Box3.csv", mode="w", newline="")
file4 = open("Box4.csv", mode="w", newline="")

for j in range(200):
    row = j
    for i in range(441):
        column = i * 4
        print(f"{SpectraMatrix.iloc[row, (column+3)]} is row {row}, column {column+3}")
        file1.write(f"{SpectraMatrix.iloc[row, column]},")
        file2.write(f"{SpectraMatrix.iloc[row, (column+1)]},")
        file3.write(f"{SpectraMatrix.iloc[row, (column+2)]},")
        file4.write(f"{SpectraMatrix.iloc[row, (column+3)]},")
    file1.write("\n")
    file2.write("\n")
    file3.write("\n")
    file4.write("\n")

file1.close()
file2.close()
file3.close()
file4.close()
