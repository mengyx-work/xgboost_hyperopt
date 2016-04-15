import sys, time
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import metrics
from wrapped_xgboost import xgboost_classifier
from utils import utils

class hyperopt_xgboost(object):

    _EARLY_STOP_RATIO = 0.1
    _MAX_OPT_COUNT = 100
    _FILENAME = 'hyperopt_xgboost_output.csv'

    def __init__(self, full_data = None, label_name = None, tuning_params = None, const_params = None, data_filename = _FILENAME, max_opt_count = _MAX_OPT_COUNT):

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

        self.train_data, self.valid_data = utils.create_validation_data(full_data, valid_frac = 0.2, dep_var_name = label_name)


        # create the space for the hyperopt from tunning_params
        self.space = {}
        for key, value in tuning_params.iteritems():
            if isinstance(value, tuple):
                self.space[key] = hp.quniform(key, value[0], value[1], value[2])
            if isinstance(value, list):
                self.space[key] = hp.choice(value)

        # load the const_params
        if const_params is not None:
            self.const_params = const_params.copy()
        else:
            self.const_params = {}
        if 'early_stopping_ratio' not in self.const_params:
            self.const_params['early_stopping_ratio'] = _EARLY_STOP_RATIO
        if 'val' not in self.const_params:
            self.const_params['val'] = True

        # xgboost model
        self.xgb_classifier = xgboost_classifier(train=self.train_data, label_name=label_name)


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
        self.xgb_classifier.fit(params = params, val = val)
        pred_res = self.xgb_classifier.predict(self.valid_data)

        best_score = self.xgb_classifier.best_score
        best_iters = self.xgb_classifier.best_iters

        fpr, tpr, thresholds = metrics.roc_curve(self.xgb_classifier.test_labels, pred_res, pos_label = 1)
        AUC_score = metrics.auc(fpr, tpr)
        time_cost = time.time() - start_time

        # all the optimizing parameters used in this run
        data_row = [params[key_name] for key_name in self.columns_name]
        data_row.append(best_score)
        data_row.append(best_iters)
        data_row.append(AUC_score)
        data_row.append(time_cost)

        # write to the data file
        df = pd.read_csv(self.data_filename, index_col=0)
        df.loc[self.iter_count] = data_row
        self.iter_count += 1
        df.to_csv(self.data_filename)

        return - AUC_score


    def hyperopt_run(self):
        # Trials object where the history of search will be stored #
        trials = Trials()
        self.columns_name = list(set(self.init_tunning_params.keys()) | set(self.const_params.keys()))

        df = pd.DataFrame(columns = self.columns_name + ['best_score', 'best_iters_num', 'auc_score', 'time_cost'])
        df.to_csv(self.data_filename)
        # start the hyperparameter optimizing
        self.best = fmin(self._objective_func, self.space, algo = tpe.suggest, trials = trials, max_evals = self.max_opt_count)

        print 'best hyperparameters: ', self.best
        print 'the optimizing is finished after {0} runs, using {1} seconds'.format(self.iter_count, time.time() - self.start_time)
