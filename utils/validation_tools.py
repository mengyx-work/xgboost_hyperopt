import pandas as pd
import numpy as np
import os, sys, time
from random import shuffle
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold


def cross_validate_model(train_df, train_label, classifier, eval_func, fold_num=2):
    
    results = []
    skf = StratifiedKFold(train_label, fold_num, shuffle=True)

    for train, test in skf:
        kfold_train = train_df.iloc[train, :]
        kfold_train_label = train_label[train]
        kfold_test = train_df.iloc[test, :]
        kfold_test_label = train_label[test]
        classifier.fit(kfold_train, kfold_train_label)
        scores = classifier.predict(kfold_test)
        result = eval_func(kfold_test_label, scores)
        results.append(result)
        
    return results


## assuming the model output is the probability of being default,
## then this probability can be used for ranking. Then using the fraction of
## default in validation data to assign the proper threshold to the prediction
def score_MCC(ground_truth, scores):
    if isinstance(ground_truth, pd.Series):
        ground_truth = ground_truth.values

    tmp_ground_truth = ground_truth[:]
    fault_frac = tmp_ground_truth.mean()
    print 'score shape:', scores.shape, 
    print 'mean of groud truth:', fault_frac
    thres_value = np.percentile(scores, 100.*(1-fault_frac), axis=0)
    print 'threshold value:', thres_value
    binary_scores = scores > thres_value
    binary_scores = binary_scores.astype(int)
    ## convert to sk-learn format
    np.place(binary_scores, binary_scores==0, -1)
    np.place(tmp_ground_truth, tmp_ground_truth==0, -1)
    #print ground_truth
    #print binary_scores
    return matthews_corrcoef(tmp_ground_truth, binary_scores)


def create_validation_index(df, valid_frac = 0.2, dep_var_name = 'dep_var'):
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


