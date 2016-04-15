import xgboost as xgb
import time, sys, os
import numpy as np
import multiprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.utils import shuffle
#from __future__ import print_function

class xgboost_classifier(object):

    _EARLY_STOPPING_ROUNDS = 100

    '''
    to properly use the xgboost_classifer repeatly with loading the data once.
    1. store the data and label columns
    2. keep using the same set for following training, only difference is the params
    '''

    def __init__(self, train = None, label_name = None, params = None):

        self.params = {}

        self.params["objective"]                = "binary:logistic"
        self.params["eta"]                      = 0.0075
        self.params["num_round"]                = 10000
        self.params["colsample_bytree"]         = 0.8
        self.params["subsample"]                = 0.8
        self.params["silent"]                   = 1
        self.params["max_depth"]                = 6
        self.params["gamma"]                    = 0
        self.params["metrics"]                  = 'auc'
        self.params['eval_metric']              = 'auc'
        self.params["seed"]                     = 100
        self.params['early_stopping_ratio']     = 0.08
        self.params['nthread']                  = multiprocessing.cpu_count()

        if params is not None:
            for key, value in params.iteritems():
                self.params[key] = value

        if train is not None:
            self.train = train
        if label_name is not None:
            self.label_name = label_name

        if train is not None and label_name is not None:
            if label_name in train.columns:
                self.train_labels = self.train[label_name]
                self.train        = self.train.drop(label_name, axis = 1)
        else:
            self.train_labels = None


    def _check_xgboost_params(self, train, label_name, params, val):

        if params is None:
            self.fit_params = self.params.copy()
        else:
            self.fit_params = self.params.copy()
            self.fit_params.update(params)

        if 'num_round' not in self.fit_params:
            raise NameError('\n Error: num_round is not defined in params.')
            sys.exit(0)

        if val is not None:
            self.fit_params['val'] = val
        elif 'val' not in self.fit_params:
            sys.stderr.write('\n by default, no early stopping evaluation.')
            self.fit_params['val'] = False

        if label_name is not None:
            self.label_name = label_name
        elif not hasattr(self, 'label_name'):
            raise ValueError('\n Error: label_name is not defined. \n')
            sys.exit(0)



    def fit(self, train = None, label_name = None, params = None, val = None):
        '''
        train given here is not the self.train, when train is missing self.train will be used
        '''
        self._check_xgboost_params(label_name, params, val)

        num_round   = self.fit_params['num_round']
        val         = self.fit_params['val']

        if train is None and self.train is None:
            raise ValueError('\n Error: train data is not defined')
            sys.exit(0)

        if train is None:
            train = self.train

        train_labels = train[self.label_name] # define the dep_var
        if self.label_name in train.columns:
            train  = train.drop(self.label_name, axis=1) # drop the dep_var in training

        # optional attributes
        self.best_score, self.best_iters = None, None
        start_time = time.time()

        if val:
            sys.stderr.write('\n####################\n train the xgboost with early stopping\n####################\n')
            # define the offset for early stopping #
            EARLY_STOP_OFFSET = int(train.shape[0] * self.fit_params['early_stopping_ratio'])
            dvalid = xgb.DMatrix(np.array(train)[:EARLY_STOP_OFFSET],
                                 label = np.array(train_labels)[:EARLY_STOP_OFFSET],
                                 missing = np.NaN)

            dtrain = xgb.DMatrix(np.array(train)[EARLY_STOP_OFFSET:],
                                 label = np.array(train_labels)[EARLY_STOP_OFFSET:],
                                 missing = np.NaN)

            self.watchlist = [(dtrain, 'train'), (dvalid, 'val')]
            self.bst = xgb.train(self.fit_params, dtrain, num_round, self.watchlist, early_stopping_rounds = self._EARLY_STOPPING_ROUNDS)
            try:
                self.best_score = self.bst.best_score
                self.best_iters = self.bst.best_iteration
            except AttributeError:
                sys.stderr.write('early sotpping is not found in this training')

        else:
            sys.stderr.write('\n####################\n train the xgboost without early stopping\n####################\n')
            dtrain = xgb.DMatrix(np.array(train), label = np.array(train_labels), missing = np.NaN)

            self.watchlist = [(dtrain, 'train')]
            self.bst = xgb.train(self.fit_params, dtrain, num_round, self.watchlist)

        print 'the xgboost fit is finished, using {} seconds'.format(time.time() - start_time)

        return self




    def cross_validate_fit(self, train = None, label_name = None, params = None, val = None, n_folds = 5):
        '''
        cross-validation fit on the entire trainign data by k-fold.
        return three arrays:
        scores, best_fit_scores, best_iter_nums

        score metric is fixed to AUC
        '''
        self._check_xgboost_params(train, label_name, params, val)
        kf = cross_validation.KFold(self.train.shape[0], n_folds = n_folds)

        # loop through the CV sets
        scores, best_fit_scores, best_iter_nums = [], [], []
        for train_index, test_index in kf:
            X = self.train[train_index]
            X_test = self.train[test_index]
            self.fit(X)
            if self.best_score is not None and self.best_iters is not None:
                print 'xgboost fit, best_score: {0}, best_iters: {1} '.format(self.best_score, self.best_iters)
                best_fit_scores.append(self.best_score)
                best_iter_nums.append(self.best_iters)

            y_prob = self.predict(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(X_test[self.label_name], y_prob, pos_label = 1)
            AUC_score = metrics.auc(fpr, tpr)
            scores.append(AUC_score)

        # record the averaged score from one iteration
        avg_score = sum(scores)/len(scores)
        score_std = 100.*np.std(np.array(scores))/avg_score
        iter_scores.append(avg_score)
        iter_std.append(score_std)

        print 'the std for scores: ' + score_std + '\n'

        for i, score, best_score, iter_num in zip(range(len(scores)), scores, best_fit_scores, best_iter_nums):
            print 'CV: {0}, AUC Score: {1}, Best Fit Score: {2}, Best Iteration Num: {3}'.format(i, score, fit_score, iter_num)

        return scores, best_fit_scores, best_iter_nums


    def predict(self, test = None):

        if test is None:
            raise ValueError('test data is not defined.')

        if self.label_name not in test.columns:
            raise ValueError('\n Error: ' + self.label_name + ' is missing in test_data')
            sys.exit(0)

        test_labels = test[self.label_name]
        test_data = test.drop(self.label_name, axis=1)
        if self.label_name in test_data.columns:
            raise ValueError('fails to drop ' + self.label_name + ' in the test data!')
            sys.exit(0)

        dtest = xgb.DMatrix(np.array(test_data), label = np.array(test_labels), missing = np.NaN)

        if self.best_iters is not None:
            y_prob = self.bst.predict(dtest, ntree_limit = self.best_iters)
        else:
            y_prob = self.bst.predict(dtest)

        return y_prob
