from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# import data as pandas data frame
train_data = pd.read_csv("/home/ooa/Belgeler/Machine Learning/HW2/Data/optdigits.tra", header=None)
test_data = pd.read_csv("/home/ooa/Belgeler/Machine Learning/HW2/Data/optdigits.tes", header=None)

# convert pandas data frame to ndarray
train_data = train_data.values 
test_data = test_data.values

X = np.asarray(train_data[:,:64], dtype=np.float64)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Weighted Euclidean Distance
def weightedDist(a,b,w):
    q = a-b
    return np.sqrt((w*q*q).sum())

kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
predicted_array = kmeans.predict(test_data[:,:64])
kmeans.labels_
kmeans.cluster_centers_

cent = 10
y = X[np.random.choice(X.shape[0], cent, replace=False)]
print(y)

def kmmeans(train, center, k):
    label = np.zeros(len(train))
    for i in range(len(train)):
        distance = dist(train[i], center)
        index_label = np.argmin(distance)
        new_center = (center[index_label] + train[i])/2
        center[index_label] = new_center
        label[i] = index_label 
    return label

ll = kmmeans(X, kmeans.cluster_centers_, 10)
print(ll)
