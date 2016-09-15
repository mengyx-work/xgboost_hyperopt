import xgboost as xgb
import operator, tempfile
import time, sys, os
import pickle, warnings
import numpy as np
import pandas as pd
import multiprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
#from __future__ import print_function

class xgboost_classifier(object):

    _EARLY_STOPPING_ROUNDS = 200

    '''
    to properly use the xgboost_classifer repeatly with loading the data once.
    1. store the data and label_name
    2. keep using the same set for following training, if the training data is not given
    3. cross_validate_fit uses fit and predict functions, so it doesn't remove dep_var column
    '''

    def __init__(self, train = None, label_name = None, params = None, use_weights = False, use_scale_pos_weight = False,  model_file = './current_xgboost_model'):

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
        #self.params['nthread']                  = 2 * multiprocessing.cpu_count()
        #self.params['nthread']                  = 8

        self.model_file_name        = model_file
        self.use_weights            = use_weights
        self.use_scale_pos_weight   = use_scale_pos_weight

        if params is not None:
            ## try to load use_weights from params
            param_use_weights = params.pop('use_weights', False)
            self.use_weights = any([param_use_weights, self.use_weights])

            for key, value in params.iteritems():
                self.params[key] = value

        if train is not None:
            self.train = train
        if label_name is not None:
            self.label_name = label_name


    def _check_xgboost_params(self, label_name, params, val):
        '''
        helper function to check the model parameters before
        training the model
        '''
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


    def _validate_training_data(self, train, split_train = True):
        '''
        helper function to 
        1. validate train and label_name are legit
        2. split train into train (a separate DataFrame without label)
        and the train_label Series
        '''
        if train is None and self.train is None:
            raise ValueError('\n Error: train data is not defined')
            sys.exit(0)

        if train is None:
            if split_train:
                train_labels = self.train[self.label_name]
                train        = self.train.drop(self.label_name, axis=1)
                return train, train_labels
            else:
                return self.train

        if self.label_name not in train.columns:
            raise ValueError('\n Error: expected the dep_var named ' + self.label_name + ' is not in the training data.')
            sys.exit(0)

        if split_train:
            train_labels = train[self.label_name]
            ## a new/different train is created
            train  = train.drop(self.label_name, axis=1)
            return train, train_labels
        else:
            return train


    def get_feature_impoartance(self):
        if not hasattr(self, 'bst'):
            raise ValueError('no booster is found.')
        return self.bst.get_fscore()


    def load_model_from_file(self, model_file = None):
        if not os.path.isfile(self.model_file_name) and model_file == None:
            raise ValueError('model file is missing.')

        if model_file is not None:
            ## update self.model_file_name
            self.model_file_name = model_file

        if not os.path.isfile(self.model_file_name):
            raise ValueError('model file is not found in {}'.format(self.model_file_name))

        self.bst = xgb.Booster()
        self.bst.load_model(self.model_file_name)


    def _ceate_feature_map(self, features, fea_map_file):
        with open(fea_map_file, 'w') as outfile:
            i = 0
            for feat in features:
                outfile.write('{0}\t{1}\tq\n'.format(i, feat))
                i = i + 1


    def _create_feature_importance_map(self, fea_map_file):
        importance = self.bst.get_fscore(fmap = fea_map_file)
        importance = sorted(importance.items(), key = operator.itemgetter(1))

        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['norm_fscore'] = df['fscore'] / df['fscore'].sum()
        df.to_csv(self.model_file_name + '_feature_importance.csv')
        os.remove(fea_map_file)


    def fit(self, train = None, label_name = None, use_weights = False, use_scale_pos_weight = False, params = None, val = None):
        '''
        train given here is not the self.train, when train is missing self.train will be used
        Design principle
        1. the full training data is imported with dep_var_name given at the same time
        2. the full training data as a reference is passed around; for example in the 
        cross_validate_fit. 
        3. the train_labels is separated from train by using the function _validate_training_data
        in the fit function only.
        '''
        ## load the params
        self._check_xgboost_params(label_name, params, val)

        ## combine the initial param with current use_weights
        use_weights = any([use_weights, self.use_weights])
        use_scale_pos_weight = any([self.use_scale_pos_weight, use_scale_pos_weight])

        if use_scale_pos_weight:
            scale_pos_weight = 1. * np.sum(train[self.label_name] == 0) / np.sum(train[self.label_name] == 1)
            self.fit_params['scale_pos_weight'] = scale_pos_weight

        
        if use_weights:
            weights = self._create_weight_by_label(train[self.label_name])
            ## check the weight dimension with train
            if len(weights) != train.shape[0]:
                sys.exit('the weights dimension {} does not match train {}, abort...'.format(len(weights), train.shape[0]))

        ## split the train_labels from train
        train, train_labels = self._validate_training_data(train, split_train = True)
        tmpfile = tempfile.NamedTemporaryFile(mode='w', prefix='xgbFeaMap_',delete=False)
        self.fea_map_file = tmpfile.name
        self._ceate_feature_map(train.columns, self.fea_map_file)

        num_round   = self.fit_params['num_round']
        val         = self.fit_params['val']

        # optional attributes
        self.best_score, self.best_iters = None, None
        start_time = time.time()

        if val:
            sys.stderr.write('\n####################\n train the xgboost with early stopping\n####################\n')
            # define the offset for early stopping #
            EARLY_STOP_OFFSET = int(train.shape[0] * self.fit_params['early_stopping_ratio'])

            if not use_weights:
                dvalid = xgb.DMatrix(np.array(train)[:EARLY_STOP_OFFSET],
                                     label = np.array(train_labels)[:EARLY_STOP_OFFSET],
                                     missing = np.NaN)

                dtrain = xgb.DMatrix(np.array(train)[EARLY_STOP_OFFSET:],
                                     label = np.array(train_labels)[EARLY_STOP_OFFSET:],
                                     missing = np.NaN)
            else:
                print 'weights are used for xgboost training data...'
                dvalid = xgb.DMatrix(np.array(train)[:EARLY_STOP_OFFSET],
                                     label = np.array(train_labels)[:EARLY_STOP_OFFSET],
                                     missing = np.NaN, weight=weights[:EARLY_STOP_OFFSET])

                dtrain = xgb.DMatrix(np.array(train)[EARLY_STOP_OFFSET:],
                                     label = np.array(train_labels)[EARLY_STOP_OFFSET:],
                                     missing = np.NaN, weight=weights[EARLY_STOP_OFFSET:])


            self.watchlist = [(dtrain, 'train'), (dvalid, 'val')]
            self.bst = xgb.train(self.fit_params, dtrain, num_round, self.watchlist, early_stopping_rounds = self._EARLY_STOPPING_ROUNDS)
            try:
                self.best_score = self.bst.best_score
                self.best_iters = self.bst.best_iteration
            except AttributeError:
                sys.stderr.write('early sotpping is not found in this training')

        else:
            sys.stderr.write('\n####################\n train the xgboost without early stopping\n####################\n')
            if not use_weights:
                dtrain = xgb.DMatrix(np.array(train), label = np.array(train_labels), missing = np.NaN)
            else:
                print 'weights are used for xgboost training data...'
                dtrain = xgb.DMatrix(np.array(train), label = np.array(train_labels), missing = np.NaN, weight=weights)

            self.watchlist = [(dtrain, 'train')]
            self.bst = xgb.train(self.fit_params, dtrain, num_round, self.watchlist)
        
        self._create_feature_importance_map(self.fea_map_file)

        self.bst.save_model(self.model_file_name)
        print 'the xgboost fit is finished by using {} seconds, saved into {}'.format((time.time() - start_time), self.model_file_name)

        return self


    def _create_weight_by_label(self, label):
        if isinstance(label, pd.Series):
            label = label.values
        scale_pos_weight = 1. * np.sum(label == 0) / np.sum(label == 1)
        print 'scale_pos_weight:', scale_pos_weight

        init_weight = np.copy(label)

        np.place(init_weight, init_weight==1, scale_pos_weight)
        np.place(init_weight, init_weight==0, 1.)

        init_weight = init_weight.astype(float)
        return init_weight


    def cross_validate_fit(self, eval_func, train = None, label_name = None, use_weights=False, params = None, val = None, n_folds = 5):
        
        self._check_xgboost_params(label_name, params, val)
        train = self._validate_training_data(train, split_train = False)
        results = []
        #print 'label name:', self.label_name
        skf = StratifiedKFold(train[self.label_name], n_folds, shuffle=True)
        #print 'train shape:', train.shape

        for train_index, test_index in skf:
            kfold_train = train.iloc[train_index, :]
            kfold_test  = train.iloc[test_index, :]
            kfold_test_label = kfold_test[self.label_name]
            self.fit(train = kfold_train, use_weights = use_weights)
            scores = self.predict(kfold_test)
            result = eval_func(kfold_test_label, scores)
            results.append(result)
        
        return results



    def predict(self, test = None):

        if test is None:
            raise ValueError('test data is not defined.')

        ## test data may not contain the label_name column
        if self.label_name in test.columns:
            #raise ValueError('\n Error: ' + self.label_name + ' is missing in test_data')
            #sys.exit(0)
            test_labels = test[self.label_name]
            test_data = test.drop(self.label_name, axis=1)
            dtest = xgb.DMatrix(np.array(test_data), label = np.array(test_labels), missing = np.NaN)
        else:
            warnings.warn('in the xgboost prediction, test data does not contain label column ' + self.label_name)
            dtest = xgb.DMatrix(np.array(test), missing = np.NaN)

        if hasattr(self, 'best_iters') and self.best_iters is not None:
            y_prob = self.bst.predict(dtest, ntree_limit = self.best_iters)
        else:
            y_prob = self.bst.predict(dtest)

        return y_prob



'''
    def cross_validate_fit(self, train = None, label_name = None, params = None, val = None, n_folds = 5):

        self._check_xgboost_params(label_name, params, val)
        train = self._validate_training_data(train, split_train = False)
        # create k-fold validation index group
        kf = cross_validation.KFold(train.shape[0], n_folds = n_folds)

        # loop through the CV sets
        scores, best_fit_scores, best_iter_nums = [], [], []
        for train_index, test_index in kf:
            X = train.iloc[train_index, ]
            X_test = train.iloc[test_index, ]
            self.fit(X)
            if self.best_score is not None and self.best_iters is not None:
                print 'xgboost cross-validate fit, best_score: {0}, best_iters: {1} '.format(self.best_score, self.best_iters)
                best_fit_scores.append(self.best_score)
                best_iter_nums.append(self.best_iters)

            y_prob = self.predict(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(X_test[self.label_name], y_prob, pos_label = 1)
            AUC_score = metrics.auc(fpr, tpr)
            scores.append(AUC_score)

        # record the averaged score from one iteration
        avg_score = sum(scores)/len(scores)
        score_std = np.std(np.array(scores))

        # print out the results
        print 'AUC scores, std: ' + str(score_std) + ', average score: ' + str(avg_score) + '\n'
        for i, score, best_score, iter_num in zip(range(len(scores)), scores, best_fit_scores, best_iter_nums):
            print 'CV: {0}, AUC Score: {1}, Best Fit Score: {2}, Best Iteration Num: {3}'.format(i, score, best_score, iter_num)

        return scores, best_fit_scores, best_iter_nums
'''

