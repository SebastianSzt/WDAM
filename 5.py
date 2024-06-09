# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:06:15 2024

@author: pk
"""

#orange biolab

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, tree, metrics
from sklearn.model_selection import 
from sklearn.metrics import precision_score, recall_score


iris = datasets.load_iris()

iris.data

iris.features_names
plt.scatter(
    iris.data.T[2], 
    iris.data.T[1],
    c = iris.target
    )

clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

predicted = clf.predict(iris.data)

#iris.target[60]

predicted

print(metrics.classification_report(iris.target, predicted, target_names = iris.target_names))

CM = metrics.confusion_matrix(iris.target, predicted)
CM

kf = KFold(n_splits=3)
sidxs = np.arange(iris.data.shape[0])
np.random.shuffle(sidxs)

for i, (train_index, test_index) in enumerate(kf.split(X)):
     print(f"Fold {i}:")
     print(f" shape = %i" % (len(train_index)))
     
     X_train = iris.data[sidxs]
     X_test = iris.data[sidxs]
     y_train = iris.target[sidxs]
     y_test = iris.target[sidxs]
     
     clf = tree.DecisionTreeClassifier()
     clf.fit(X_train, y_train)
     y_pred = clf.predict(X_test)
     acc_score = metrics.accuracy_score(y_test, y_pred)
     print("ACC = %f" % acc_score)
     print(metrics.classification_report(y_test, y_pred))
     
     
     
     
     
     
     