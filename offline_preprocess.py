#!/usr/bin/env python3

import utils
import csv
import os
import sys
import numpy as np
import time

def diff(d):
    """order 1 difference for non-decreasing timeseries"""
    diff_1 = np.diff(d)  # order 1 difference
    if np.all(diff_1 >= 0):  # non-decreasing ts
        return np.array([diff_1[0]] + diff_1.tolist())
    else:
        return d


def scale(d):
    """"min-max scaler"""
    return minmax(d)


def smoothen(d, alpha=0.5):
    """expoenential moving average
    TODO: to be vectorized for better performance
    """
    alpha_rev = 1 - alpha
    res = [d[0]]
    for i in d[1:]:
        res.append(alpha*i + alpha_rev * res[-1])
    return np.array(res)


def preprocess(d):
    """pre-processing chain"""
    return smoothen(scale(diff(d)))


dev_dir = get_device_yang_dir()
merged_csv = "merged_190519_700_leaf7.csv"

tstp, data = data_loader(merged_csv, scale=False)

ft_name = np.array(get_feature_names_bis(merged_csv))
ft_name = ft_name[1:]  # remove the name for timestamps

# removing cols containing nan and inf values
inval_col = np.where(np.any(np.isnan(data), axis=0))
data = np.delete(data, inval_col, axis=1)
ft_name = np.delete(ft_name, inval_col)
inval_col = np.where(np.any(np.isinf(data), axis=0))
data = np.delete(data, inval_col, axis=1)
ft_name = np.delete(ft_name, inval_col)
print("Data shape after nan inf removal: " + str(data.shape))

start = time.time()
for n_col in range(data.shape[1]):
    data[:, n_col] = preprocess(data[:, n_col])
end = time.time()
print("Processed %d times of %d in length in %.4f sec." % 
      (data.shape[1], data.shape[0], end-start))

data = np.hstack((np.array(tstp).reshape((-1,1)), data))
if "synthetic" in get_device():
    data = np.hstack((data, events))
    
np.savetxt(preprocessed_csv,
           data,
           fmt="%.9f",
           delimiter=',',
           header=','.join(ft_name),
           comments='')

