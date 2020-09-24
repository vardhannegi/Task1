from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from inputDatagen import circuitgen as crg
import pandas as pd
import tensorflow as tf
import time
import numpy as np

n = 2
m = 10000

dataClass = crg(total_layer = n,total_instance = m)
data = dataClass.getInputData()
##print(data)

df = pd.DataFrame(data)
X = np.array(df.drop(['epsilon'], 1))
y = np.array(df['epsilon'])

X = preprocessing.scale(X)
y = np.array(df['epsilon'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = svm.SVR() #LinearRegression()
clf.fit(X_train, y_train)


confidence = clf.score(X_test, y_test)
print(confidence)


