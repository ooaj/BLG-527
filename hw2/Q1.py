from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from copy import deepcopy

# import data as pandas data frame
train_data = pd.read_csv("/home/ooa/Belgeler/Machine Learning/HW2/Data/optdigits.tra", header=None)
test_data = pd.read_csv("/home/ooa/Belgeler/Machine Learning/HW2/Data/optdigits.tes", header=None)

# this will be used for label display later on
train_data_pd = pd.read_csv("/home/ooa/Belgeler/Machine Learning/HW2/Data/optdigits.tra", header=None)

# convert pandas data frame to ndarray
train_data = train_data.values 
test_data = test_data.values

# Integer to float, exclude the last column because it is the predetermined labels
X = np.asarray(train_data[:,:64], dtype=np.float64)

# Euclidean Distance
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Weighted Euclidean Distance
def weightedDist(a,b,w):
    q = a-b
    return np.sqrt((w*q*q).sum())

# Scikit implementation for Comparison
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
#predicted_array = kmeans.predict(test_data[:,:64])
#kmeans.labels_
#kmeans.cluster_centers_


# cent corresponds to K in our value, i.e how many clustering groups we will have
cent = 10
'''Random centroid location picker
Uncomment below if you dont want to compare with the scikit results, and use your own centroid values
'''
#y = X[np.random.choice(X.shape[0], cent, replace=False)]
#print(y)
center = kmeans.cluster_centers_
# k-Means clustering
initial_center = np.zeros(center.shape)
label = np.zeros(len(X))
recon_error = dist(center, initial_center , None)
# k-Means clustering
while recon_error != 0:
    
    for i in range(len(X)):
        distance = dist(X[i], center)
        index_label = np.argmin(distance)
        label[i] = index_label
        
    initial_center = deepcopy(center)
    
    for j in range(len(X)):
        new_center = (center[index_label] + X[j])/2
        center[index_label] = new_center
   
    recon_error = dist(center, initial_center , None)

# Comparison of our results with Scikit
print("Comparison of our algorithm with scikit =", kmeans.labels_ == label)

print("Predicted labels are", label)

# Changing the last column of the original data with the newly clustered labels
train_data_pd[64] = label.astype(int)
print(train_data_pd)

