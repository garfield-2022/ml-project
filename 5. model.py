#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:55:21 2023

@author: yingfan
"""

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

data_path = 'creditcard.csv'
df = pd.read_csv(data_path)

# Next we will split the new dataset into Features and Target
X = df.drop(columns="Class", axis=1)
y = df["Class"]

# Split the new dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=0)


# fit the scaler on the entire data frame so that it standardizes all of the data in the same way. 
scaler = StandardScaler()
scaler.fit(X)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = make_pipeline(SMOTE(random_state=0), LogisticRegression())

cv_results = cross_validate(model, X_train, y_train, scoring="balanced_accuracy",
    return_train_score=True, return_estimator=True, n_jobs=-1)

pickle.dump(cv_results["estimator"][0], open('model.pkl', 'wb'))
