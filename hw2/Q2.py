from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# import data as pandas data frame, change according to your path
train_data = pd.read_csv("/home/ooa/Belgeler/Machine Learning/HW2/Data/optdigits.tra", header=None)
test_data = pd.read_csv("/home/ooa/Belgeler/Machine Learning/HW2/Data/optdigits.tes", header=None)

# convert pandas data frame to ndarray
train_data = train_data.values 
test_data = test_data.values

# Integer to float, exclude the last column because it is the predetermined labels
X = np.asarray(train_data[:,:64], dtype=np.float64)
y = np.asarray(test_data[:,:64], dtype=np.float64)

# random test to print the digit 7 from the data for data visualization
deneme_7 = train_data[30, :64]
deneme_7 = np.reshape(deneme_7, (8,8))
plt.imshow(deneme_7, interpolation='Nearest')
plt.show()


# Euclidean Distance 
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

# Weighted Euclidean Distance
def wdist(a,b,w):
    q = a-b
    return np.sqrt(np.sum(q*np.var(q, axis=1)))

# Neighbours
def neighbours(train, test, label, k):
    distance_matrix = []
    for i in range(len(train)):
        distance = dist(test, train[i]) # distance between data points
        distance_matrix.append((distance, label[i]))
    distance_matrix.sort(key=lambda x: x[0]) # sort the distance from lowest to highest
    neighbors = distance_matrix[:k] # pick the lowest
    return neighbors

# Iterate the function all over the data
correct = 0
label = np.zeros(len(test_data))
for j in range(len(test_data)):
    k1 = 1
    neighbors = neighbours(train_data, test_data[j], train_data[:,64], k1)
    neighborss = np.array(neighbors)
    (values,counts) = np.unique(neighborss,return_counts=True)
    ind=np.argmax(counts)
    final_label = (values[ind])
    label[j] = final_label
    
    # Get the number of times we have correctly labeled the data (later used for accuracy)
    if float(test_data[:,64][j]) == final_label:
        correct += 1
    
    print('loop number =', j,', test data label =', test_data[:,64][j], ', knn, distance + labels =', neighbors, ', label according to knn =', final_label)

# The accuracy of kNN
accuracy = correct/len(test_data)*100
#print("the final accuracy is =", accuracy)

label_int = label.astype(int)
#dff = pd.DataFrame({'kNN (k=1) labels':label_int})
#print(dff)

# The results with the built-in function from Scikit
knn = KNeighborsClassifier()
knn.fit(train_data[:,:64], train_data[:,64]) 
# change uniform to distance for weighted euclidean (ref:)
KNeighborsClassifier(metric='euclidean',n_neighbors=1, weights='uniform')
knn_predict = knn.predict(test_data[:,:64])
print("Predictions from the classifier:")
print(knn.predict(test_data[:,:64]))
print("Target values:")
print(test_data[:,64])



kmeans = KMeans(n_clusters=30, random_state=0).fit(X)
predicted_array = kmeans.predict(y)

# Accuracy for k-Means
correctt = 0
for u in range(len(test_data)):
    if predicted_array[u] == test_data[:,64][u]:
        correctt += 1
    else:
        correctt += 0
        
accuracyy = correctt/len(test_data)*100




df = pd.DataFrame({'Original Labels':test_data[:,64], 'Labels according to kNN':label_int, 'Labels according to clustering':predicted_array})

print(df)

print("the final accuracy is =", accuracy)
print("Accuracy of k-means is =", accuracyy)