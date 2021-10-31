#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import pickle

# parameters selected while tuning
max_depth=15
min_samples_leaf = 1
n_estimators=30
bootstrap=False
random_state=42
output_file = 'rf_model.bin'

# data preparation
df_cols = ['satisfaction_level','last_eval','num_projects','avg_monthly_hrs','time_spent_at_company',
           'work_accident','quit','promotion_last_5_yrs','job_category','salary_group']

df = pd.read_csv('data/HR_comma_sep.csv', names=df_cols, skiprows=1)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['quit']
y_test = df_test['quit']

del df_train['quit']
del df_test['quit']


# function to train the model
def train(df_train, y_train, n_estimators, max_depth, min_samples_leaf, bootstrap, random_state):
    
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      bootstrap=bootstrap,
                                      random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    return dv, rf_model

# function to predict using the model and DictVectorizer
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


dv, rf_model = train(df_train, y_train, n_estimators, max_depth, min_samples_leaf, bootstrap, random_state)

# saving the DictVectorizer and RandomForest model to the same file
with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, rf_model), f_out)

