import sys, time, os
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import metrics
from wrapped_xgboost import xgboost_classifier
from utils import validation_tools
from utils.utils_functions import utils_functions

class hyperopt_xgboost(object):

    _EARLY_STOP_RATIO = 0.1
    _MAX_OPT_COUNT = 100
    _FILENAME = 'hyperopt_xgboost_output.csv'

    def __init__(self, full_data = None, label_name = None, tuning_params = None, const_params = None, data_filename = _FILENAME, max_opt_count = _MAX_OPT_COUNT, crosValid_mode=False, fold_num=3):

        if tuning_params is None:
            raise ValueError('tunning parameters are not given')
            sys.exit(0)

        if full_data is None or label_name is None:
            raise ValueError('full_data or label_name is missing')
            sys.exit(0)

        self.data_filename       = data_filename
        self.max_opt_count       = max_opt_count
        self.init_tunning_params = tuning_params
        self.start_time          = time.time()
        self.iter_count          = 0
        self.label_name          = label_name
        ## params for crosValid mode
        self.crosValid_mode      = crosValid_mode
        self.fold_num            = fold_num

        ## create the space for the hyperopt from tunning_params
        self.space = {}
        for key, value in tuning_params.iteritems():
            if isinstance(value, tuple):
                self.space[key] = hp.quniform(key, value[0], value[1], value[2])
            if isinstance(value, list):
                self.space[key] = hp.choice(value)

        ## load the const_params
        if const_params is not None:
            self.const_params = const_params.copy()
        else:
            self.const_params = {}
        if 'early_stopping_ratio' not in self.const_params:
            self.const_params['early_stopping_ratio'] = _EARLY_STOP_RATIO
        if 'val' not in self.const_params:
            self.const_params['val'] = True

        ## wrapper_xgboost provides additional function to do cross validation on a training data
        ## the crosValid_mode support this method
        ## Otherwise, it uses the standard way to build and train xgboost from wrapped_xgboost
        if not self.crosValid_mode:
            self.train_data, self.valid_data = utils_functions.create_validation_data(full_data, valid_frac=0.2, dep_var_name=self.label_name)
            # xgboost model
            self.xgb_classifier = xgboost_classifier(train=self.train_data, label_name=self.label_name)
        else:
            self.full_data = full_data



    def _objective_func(self, tunning_params):

        # collecte params from the object
        params = {}
        if self.const_params is not None:
            params = self.const_params.copy()
        params.update(tunning_params)

        # the hyperopt messed up the function argument type, it is corrected here
        for key in self.init_tunning_params:
            if isinstance(self.init_tunning_params[key][0], (int, long)):
                params[key] = int(tunning_params[key])
            elif isinstance(self.init_tunning_params[key][0], float):
                params[key] = float(tunning_params[key])

        if 'val' in params:
            val = params['val']
        else:
            val = True

        start_time = time.time()
        # all the optimizing parameters used in this run
        data_row = [params[key_name] if key_name in params.keys() else 'NaN' for key_name in self.columns_name]

        if self.crosValid_mode:
            self.xgb_classifier = xgboost_classifier(label_name=self.label_name, params=params)
            results = self.xgb_classifier.cross_validate_fit(validation_tools.score_MCC, self.full_data, n_folds=self.fold_num)
            avg_score =  np.average(results)
            score_std = np.std(results)
            data_row.append(avg_score)
            data_row.append(score_std)
        else:
            self.xgb_classifier.fit(params = params, val = val)
            pred_res = self.xgb_classifier.predict(self.valid_data)
            ## assuming the xgb_classifier will provide both score and iter_num
            best_score = self.xgb_classifier.best_score
            best_iters = self.xgb_classifier.best_iters
            ## use AUC as the metric
            fpr, tpr, thresholds = metrics.roc_curve(self.xgb_classifier.test_labels, pred_res, pos_label = 1)
            AUC_score = metrics.auc(fpr, tpr)
            ## write results into csv file 
            data_row.append(best_score)
            data_row.append(best_iters)
            data_row.append(AUC_score)

        ## write the time cost to file
        time_cost = time.time() - start_time
        data_row.append(time_cost)

        # write to the data file
        df = pd.read_csv(self.data_filename, index_col=0)
        df.loc[self.iter_count] = data_row
        self.iter_count += 1
        df.to_csv(self.data_filename)

        #return - AUC_score
        return -avg_score


    def hyperopt_run(self):
        # Trials object where the history of search will be stored #
        trials = Trials()
        self.columns_name = list(set(self.init_tunning_params.keys()) | set(self.const_params.keys()))
        print 'param columns:', self.columns_name
        print len(self.columns_name)
        if not self.crosValid_mode:
            df = pd.DataFrame(columns = self.columns_name + ['best_score', 'best_iters_num', 'auc_score', 'time_cost'])
        else:
            df = pd.DataFrame(columns = self.columns_name + ['avg_score', 'score_std', 'time_cost'])

        df.to_csv(self.data_filename)
        # start the hyperparameter optimizing
        self.best = fmin(self._objective_func, self.space, algo = tpe.suggest, trials = trials, max_evals = self.max_opt_count)

        print 'best hyperparameters: ', self.best
        print 'the optimizing is finished after {0} runs, using {1} seconds'.format(self.iter_count, time.time() - self.start_time)
