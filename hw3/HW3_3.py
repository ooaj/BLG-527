import pandas as pd
import numpy as np

train_data = pd.read_csv("/home/ooa/Belgeler/git/BLG-527/hw3/Data/optdigits.tra", header=None)
test_data = pd.read_csv("/home/ooa/Belgeler/git/BLG-527/hw3/Data/optdigits.tes", header=None)

train_data = train_data.values
test_data = test_data.values

y_train = train_data[:,64]
y_test = test_data[:,64]

X_train = train_data[:,:64]
X_test = test_data[:,:64]

# Update features.
X_train[X_train < 6 ] = 0
X_train[X_train >= 6] = 1