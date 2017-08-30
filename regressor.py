import cPickle
import json
from math import sqrt
from os.path import isfile

import numpy as np
import pandas as pd
import xgboost as xgb
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def split_data(data):
    cases = data['case_id'].unique()

    cases_train_point = int(len(cases) * 0.8)

    train_cases = cases[:cases_train_point]

    ids = []
    for i in range(0, len(data)):
        ids.append(data['case_id'][i] in train_cases)

    train_data = data[ids]
    test_data = data[np.invert(ids)]
    return train_data, test_data
    
def linear(filename):
    train_data, test_data, original_test_data = prep_data(filename)
    lm = LinearRegression(fit_intercept=True)
    y = train_data['remaining_time']
    train_data = train_data.drop('remaining_time', 1)

    print 'training'
    lm.fit(train_data, y)
    with open('linearregression_'+filename+'.pkl', 'wb') as fid:
        cPickle.dump(lm, fid)

    original_test_data['prediction'] = lm.predict(test_data)

    write_pandas_to_csv(original_test_data, filename)
    rms = sqrt(mean_squared_error(original_test_data['remaining_time'], original_test_data['prediction']))
    print rms/3600
    return "done"

def randomforestregression(filename):
    train_data, test_data, original_test_data = prep_data(filename)
    rf = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
    y = train_data['remaining_time']
    train_data = train_data.drop('remaining_time', 1)
    rf.fit(train_data, y)
    with open('randomforestregression'+filename+'.pkl', 'wb') as fid:
        cPickle.dump(rf, fid)

    original_test_data['prediction'] = rf.predict(test_data)

    write_pandas_to_csv(original_test_data, filename)
    rms = sqrt(mean_squared_error(original_test_data['remaining_time'], original_test_data['prediction']))
    print rms/3600
    return "done"

def xgboost(filename):
    train_data, test_data, original_test_data = prep_data(filename)
    clf = xgb.XGBRegressor(n_estimators=2000, max_depth=10)
    y = train_data['remaining_time']
    train_data = train_data.drop('remaining_time', 1)
    clf.fit(train_data, y)
    with open('xgboost'+filename+'.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

    original_test_data['prediction'] = clf.predict(test_data)

    write_pandas_to_csv(original_test_data, filename)
    rms = sqrt(mean_squared_error(original_test_data['remaining_time'], original_test_data['prediction']))
    print rms/3600
    return "done"

def prep_data(filename):
    df = pd.read_csv(filepath_or_buffer=filename, header=0)

    train_data, test_data = split_data(df)

    train_data = train_data.drop('case_id', 1)
    original_test_data = test_data
    test_data = test_data.drop('case_id', 1)

    test_data = test_data.drop('remaining_time', 1)

    return train_data, test_data, original_test_data

def to_return_data(data):
    new_data = pd.DataFrame(index=range(0, len(data)), columns=['case_id', 'remaining_time', 'prediction'])
    new_data['case_id'] = data['case_id']
    new_data['remaining_time'] = data['remaining_time']
    new_data['prediction'] = data['prediction']

    return new_data

def write_pandas_to_csv(df, filename):    
    df.to_csv("results_freq_"+filename,sep=',',mode='w+', index=False)
    return filename

print linear("freq_encode_bpi2012_data.csv")
print randomforestregression("freq_encode_bpi2012_data.csv")
print xgboost("freq_encode_bpi2012_data.csv")