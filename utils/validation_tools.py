import pandas as pd
import numpy as np
import os, sys, time

def create_validation_index(self, df, valid_frac = 0.2, dep_var_name = 'dep_var'):
  valid_index = []
  train_index = []
  index_series = df[dep_var_name]
  grouped_index = index_series.groupby(index_series)

  for name, group in grouped_index:
    index_length = int(valid_frac * group.shape[0])
    valid_index.extend(group[0:index_length].index.tolist())
    train_index.extend(group[index_length:].index.tolist())

  # shuffle the training and test data in place
  shuffle(train_index)
  shuffle(valid_index)

  return  train_index, valid_index 


