import pandas as pd
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

deneme_7 = test_data[1, :64]
deneme_7 = np.reshape(deneme_7, (8,8))
plt.imshow(deneme_7, interpolation='Nearest')
plt.show()

clf_gini = DecisionTreeClassifier(max_depth=3, max_features=2)
clf_gini.fit(X_train, y_train)

y_pred_test = clf_gini.predict(X_test)
y_pred_train = clf_gini.predict(X_train)

print ("Test accuracy is ", accuracy_score(y_test,y_pred_test)*100)
print ("Train accuracy is ", accuracy_score(y_train,y_pred_train)*100)

from sklearn.model_selection import GridSearchCV
param_grid = [
{'max_depth': [3, 10, 30, 40, 50], 'max_features': [2, 4, 6, 8, 10, 12]},
{'criterion': ['entropy'], 'max_depth': [3, 10, 30, 40, 50], 'max_features': [2, 4, 6, 8, 10, 12]},
]
clf_gini_2 = DecisionTreeClassifier()
grid_search = GridSearchCV(clf_gini_2, param_grid, cv=5,
scoring='accuracy')
grid_search.fit(X_train, y_train)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


clf_gini_3 = grid_search.best_estimator_
clf_gini_3.fit(X_train, y_train)

y_pred_test_3 = clf_gini_3.predict(X_test)
y_pred_train_3 = clf_gini_3.predict(X_train)

print ("Test accuracy is ", accuracy_score(y_test,y_pred_test_3)*100)
print ("Train accuracy is ", accuracy_score(y_train,y_pred_train_3)*100)

end = time.time()
print(end - start)