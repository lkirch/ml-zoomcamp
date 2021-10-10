#!/usr/bin/env python
# coding: utf-8

import pickle

# import the model
with open('model1.bin', 'rb') as f_in, open('dv.bin', 'rb') as f_in2:
    model = pickle.load(f_in)
    dv = pickle.load(f_in2)
f_in.close(), f_in2.close()

# set up a customer
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

print('Scoring customer...')

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]
churn = y_pred >= 0.5

print('The probability of this custmer churning is {0}'.format(y_pred))
print('The customer is likely to churn: {0}'.format(churn))
