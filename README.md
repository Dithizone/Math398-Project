# PNNL-Project
  Welcome to the GitHub repository for the PNNL Project at CSU Channel Islands! Quite a few files have been generated so far and this README will serve as an introduction to what they are.
  
  First, **SpectraMatrix.csv** is the data provided by Dr. Aaron Luttman, with 200 rows corresponding to radiation energies and 1764 columns corresponding to each sensor in four 21-by-21 arrays (21x21x4=1764), where each "box" of 21-by-21 contains a different radiation source &mdash; Cobalt-60, Cesium-137, Technetium-99, and the KUT background &mdash; at its center. The file **SpectraMatrixWithXAxis.csv** is identical to SpectraMatrix.csv, except that the first column is x-values from 0.05 to 2.0 MeV for ease of graphing.
  
  The set of text files (**0147data.txt**, **0662data.txt**, **1332data.txt**, and **1764data.txt**) are the rows in SpectraMatrix.csv for the energy peaks at 0.147, 0.662, 1.332, and 1.764 MeV arranged as a column to graph in Excel. These peaks were selected because they're the signatures for the three radionuclides, and were used to visualize the periodic nature of signal intensities as we progress from column to column and sensor to sensor. The fact that they all cycle every 84 rather than 21 columns and all reach their highest intensity around column 884 demonstrated that the columns were progressing through each box then moving to the next sensor, rather than moving through every sensor before progressing to the next box. In other words, Box 1 is composed of columns 1, 5, 9, 13... and Box 3 is of columns 3, 7, 11, 15, and so on. This can be seen in **RadionuclideSeparation.ipynb**.
  
  The first attempt at determining the box identities was **PNNL_Analysis.ipynb**, which assumed Box 1 was columns 1 through 441, Box 2 was columns 442 to 884, etc., and amusingly seems to have iterated through just Technetium-99 and KUT background spectra, giving the illusion that all the data was identical. This Jupyter notebook is now obsolete and should either be repurposed or removed.
  
  **RadionuclideDataSlicer.py** is a short script which iterates through SpectraMatrix.csv and separates each column into their respective boxes. These boxes are **Box1.csv**, **Box2.csv**, **Box3.csv**, and **Box4.csv**. Floating point errors were introduced to several data points which is exceedingly annoying, but the introduced error doesn't change the values by more than 0.00001%, which can be seen in the Jupyter notebook **DidItSliceRightQuestionMark.ipynb**.
  
  **BoxIdentification.ipynb** proves that the radionuclides in each box are:
  - Box 1: likely the KUT background
  - Box 2: Cobalt-60
  - Box 3: Cesium-137
  - Box 4: likely Technetium-99
  In addition, at the end of BoxIdentification.ipynb, the peaks for each radionuclide are plotted as a heatmap, revealing the locations of the steel boxes.
  
  **PerformingPCA.ipynb** is the notebook for all things PCA.
  
  Things still to do:
  - Prepare slides for Luttman
  - 3D plot of PC1 vs PC2 vs PC3
  - Box2 + Box3 + Box4 -2\*Box1
  - Perform NMF
  - Perform CPT
  
