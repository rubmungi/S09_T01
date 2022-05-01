# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import scipy.stats
from scipy.stats import norm
from scipy import stats
from scipy.stats import t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from numpy.polynomial.polynomial import polyfit
from sklearn.decomposition import PCA
import matplotlib.cm as cm
vols = pd.read_csv('//home/rusi/Escritorio/rubenIT/DataSources/DelayedFlights.csv')#importem i li assignem un nom de dataframe
print(vols.info())
print(vols.describe())
print(vols.head())
vols02=vols.iloc[:,25:30].fillna(0)
print(vols02.describe(include="all"))
print(vols02.head())
vols03=vols.drop(vols.iloc[:,25:30],axis=1)
vols04 = vols03.merge(vols02, how='inner', left_index=True, right_index=True)
vols04=vols04.fillna(0)
x=vols04.drop(columns=["Distance","UniqueCarrier","TailNum","Origin","Dest","CancellationCode"])
y=vols04.Distance
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=4)
# fit the model
model = RandomForestClassifier(random_state=4)
model.fit(x_train, y_train)
# make predictions
yhat = model.predict(x_test)
# evaluate predictions
acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)