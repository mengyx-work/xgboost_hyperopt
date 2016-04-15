import os, sys, time

HOME_DIRECTORY = os.path.expanduser('~')
PROJECT_UTILS_PATH = HOME_DIRECTORY+'/Google Drive/dev/xgboost_hyperopt_wrapper/xgboost_hyperopt/utils'
sys.path.append(PROJECT_UTILS_PATH)

# suppress various warnings
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

from hyperopt_xgboost import hyperopt_xgboost
from utils import utils

# the parameters to tune by hyperopt
# parameter can be
# a. three element tuple
# b. a list of discrete values
tuning_params = {'eta':(0.001, 0.01, 0.0005), 'num_round':(5, 10, 2)}

# constant parameters as a dictionary
# avant standard xgboost params
params = {}
params["eta"]                      = 0.0075
params["subsample"]                = 0.8
params["colsample_bytree"]         = 0.8
params["num_round"]                = 10
params["max_depth"]                = 5
params["gamma"]                    = 0
params["metrics"]                  = 'auc'
params['eval_metric']              = 'auc'
params["seed"]                     = 100

params["val"]                      = False
params["early_stopping_ratio"]     = 0.2

dep_var_name = 'any_dep_var_name'
data = utils.get_mockup_data(dep_var_name)

hyperopt = hyperopt_xgboost(data, dep_var_name,
                            tuning_params = tuning_params,
                            const_params  = params,
                            data_filename = 'tmp_results.csv',
                            max_opt_count = 10)
hyperopt.hyperopt_run()
