import scipy.io as sio
import numpy as np

data = sio.loadmat("datasets/PersonGaitDataSet.mat")

X = data["X"]
Y = data["Y"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Unique Y values:", np.unique(Y))
