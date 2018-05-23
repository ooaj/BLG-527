import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import time

# import data as pandas data frame, change according to your path
start = time.time()
train_data = pd.read_csv("/home/ooa/Belgeler/git/BLG-527/hw3/Data/optdigits.tra", header=None)
test_data = pd.read_csv("/home/ooa/Belgeler/git/BLG-527/hw3/Data/optdigits.tes", header=None)

train_data = train_data.values
test_data = test_data.values

y_train = train_data[:,64]
y_test = test_data[:,64]

X_train = train_data[:,:64]
X_test = test_data[:,:64]

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

mlp = MLPClassifier(alpha = 0.000001, hidden_layer_sizes = 8)
mlp.fit(X_train, y_train)

y_pred_test = mlp.predict(X_test)
y_pred_train = mlp.predict(X_train)

print ("Test accuracy is ", accuracy_score(y_test,y_pred_test)*100)
print ("Train accuracy is ", accuracy_score(y_train,y_pred_train)*100)

param_grid = [
#{'hidden_layer_sizes': [32, 64, (64, 64), 128, (128, 128), 256]}]
{'early_stopping': [True], 'alpha': [0.0001, 0.001, 0.01], 'hidden_layer_sizes': [32, 64, (64, 64), 128, 256]}]
mlp_2 = MLPClassifier()
grid_search = GridSearchCV(mlp_2, param_grid, cv=5,
scoring='accuracy')
grid_search.fit(X_train, y_train)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

mlp_3 = grid_search.best_estimator_
mlp_3.fit(X_train, y_train)

y_pred_test_3 = mlp_3.predict(X_test)
y_pred_train_3 = mlp_3.predict(X_train)

print ("Test accuracy is ", accuracy_score(y_test,y_pred_test_3)*100)
print ("Train accuracy is ", accuracy_score(y_train,y_pred_train_3)*100)

end = time.time()
print(end - start)