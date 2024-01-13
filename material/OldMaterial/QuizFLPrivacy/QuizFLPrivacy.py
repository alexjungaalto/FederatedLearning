#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:05:47 2023

@author: alexanderjung
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import DecisionBoundaryDisplay

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target


from sklearn.datasets import load_iris


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(X[:,1])
sex = le.transform(X[:,1])

sex0 = (sex==0)
sex1 = (sex==1)

Xtrain = np.hstack((X[:,4].reshape(-1,1),X[:,5].reshape(-1,1)))

print(Xtrain.shape)

clf = DecisionTreeRegressor(max_depth=2,random_state=0).fit(Xtrain, y)



fig, ax = plt.subplots()

DecisionBoundaryDisplay.from_estimator(
        clf,
        Xtrain,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax
    )
ax.set_xlim(-0.1,0.1)
ax.set_ylim(-0.1,0.1)
ax.scatter(Xtrain[sex0,0],Xtrain[sex0,1],
             label = sex,
             c="red",
             s=15,
         )


ax.scatter(Xtrain[sex1,0],Xtrain[sex1,1],
             label = sex,
             c="blue",
             marker = "x",
             s=45,
         )

plt.show()
pred_y = clf.predict(Xtrain)
#print(np.unique(pred_y))

print("nr of different predictions :",len(np.unique(pred_y)))

for pred_val in np.unique(pred_y): 
    idx = np.isclose(pred_y,pred_val,atol = 0.001)
    print("mean of dec.region ",np.mean(Xtrain[idx,0]),np.mean(Xtrain[idx,1]))
    histvals = np.histogram(sex[idx],[-0.1,0.5,1.1])[0]
    histvals = histvals/sum(histvals)
    print("histogram of private values :",histvals)

print("normalized MSE : ", mean_squared_error(y,pred_y)/(mean_squared_error(y,np.zeros(y.shape))))

print(diabetes.feature_names[4])
print(diabetes.feature_names[5])
print(diabetes.feature_names[1])
#print(diabetes.feature_names[1])
#print(sex0)