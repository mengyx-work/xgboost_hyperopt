import os, sys, time

HOME_DIRECTORY = os.path.expanduser('~')
PROJECT_UTILS_PATH = HOME_DIRECTORY+'/Google Drive/dev/xgboost_hyperopt_wrapper/xgboost_hyperopt/utils'
sys.path.append(PROJECT_UTILS_PATH)

# suppress various warnings
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

import pandas as pd
import numpy as np
from sklearn import metrics
from wrapped_xgboost import xgboost_classifier
from utils import utils


# avant standard xgboost params
params = {}
params["eta"]                      = 0.0075
params["subsample"]                = 0.8
params["colsample_bytree"]         = 0.8
params["num_round"]                = 1500
params["max_depth"]                = 5
params["gamma"]                    = 0
params["metrics"]                  = 'auc'
params['eval_metric']              = 'auc'
params["seed"]                     = 100

params["val"]                      = True
params["early_stopping_ratio"]     = 0.2

dep_var_name = 'any_dep_var_name'
data = utils.get_mockup_data(dep_var_name)
train_data, valid_data = utils.create_validation_data(data, 0.2, dep_var_name)

xgb_clf = xgboost_classifier(train_data, dep_var_name, params = params)
xgb_clf.fit()
pred_res = xgb_clf.predict(valid_data)

fpr, tpr, thresholds = metrics.roc_curve(xgb_clf.test_labels, pred_res, pos_label = 1)
auc_score = metrics.auc(fpr, tpr)
