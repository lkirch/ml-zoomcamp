#!/usr/bin/env python
# coding: utf-8

import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://ncbjw2vjfl.execute-api.us-east-1.amazonaws.com/test/predict'

data = {'url':'https://www.cnet.com/a/img/O3eTNcOBpUKWs7ve7NMFsp6mBgU=/1200x630/2020/07/15/8577c1b7-2af0-4e3a-9384-a8cdef6fd098/fire-pits-fire-1.jpg'}

result = requests.post(url, json=data).json()
print(result)