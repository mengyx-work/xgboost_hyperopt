import os, sys, time
import yaml
import itertools

models_yaml_file = 'raw_combined_models.yml'
project_path = './'

if not os.path.exists(project_path):
    os.makedirs(project_path)

models = {}
model_index = 0

## for different random seed
def auto_fill_model_dicts(models, const_params, list_param_dict, model_type, model_index=None):
    if model_index is None:
        model_index = -1

    for key, list_param in list_param_dict.items():
        for param in list_param:
            model_params = const_params.copy()
            model_dict = {}
            model_index += 1
            model_params[key] = param
            model_dict['model_params'] = model_params
            model_dict['model_type'] = model_type
            print 'build a {} model using params: {}'.format(model_type, model_params)
            models[model_index] = model_dict



def generate_models_params_dict(models, const_params, tuning_params, model_type, model_index=None):
    if model_index is None:
        model_index = -1

    param_dict = const_params.copy()

    ## combine tuning_params with const_params 
    for key, value in tuning_params.items():
        param_dict[key] = value

    ## convert numerical and string into list
    for key, value in param_dict.items():
        if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
            param_dict[key] = [value]

    ## expand param_dict into list of params sets
    params_list = list(itertools.product(*param_dict.values()))
    
    for params in params_list:
        model_dict = {}
        model_index += 1
        model_params = {}
        ## convert params from a list into the dict
        for value, key in zip(params, param_dict.keys()):
            model_params[key] = value

        model_dict['model_params'] = model_params
        model_dict['model_type'] = model_type
        print 'build a {} model using params: {}'.format(model_type, model_params)
        models[model_index] = model_dict


#'''
params = {}
params["eta"]                      = 0.0075
params["subsample"]                = 0.8
params["colsample_bytree"]         = 0.8
params["num_round"]                = 501
params["max_depth"]                = 5
params["gamma"]                    = 0
params["metrics"]                  = 'auc'
params['eval_metric']              = 'auc'
params["seed"]                     = 999
params['verbose_eval']             = 50
## whether to use weights
params['use_base_score']           = True
params['use_weights']              = True
#params['use_scale_pos_weight']     = True
params["val"]                      = False

'''
xgb_const_params = params
list_param_dict = {}
#list_param_dict['num_round'] = [101, 201, 40l, 601]
#list_param_dict['num_round'] = [1000, 3000, 5000]
list_param_dict['max_depth'] = [2, 3, 4, 5, 6, 8, 15]
list_param_dict['colsample_bytree'] = [1., 0.8, 0.6, 0.4, 0.2]
#list_param_dict['subsample'] = [1., 0.8, 0.6]
generate_models_params_dict(models, xgb_const_params, list_param_dict, 'Xgboost', model_index)
'''


#'''
## manually add every model into the combined model
tmp_params = params.copy()
#tmp_params['max_depth'] = 14
#tmp_params['num_round'] = 601
model_index += 1
model_dict = {}
model_dict['model_type'] = 'Xgboost'
model_dict['model_params'] = tmp_params
models[model_index] = model_dict

'''
tmp_params = params.copy()
tmp_params['max_depth'] = 12
tmp_params['num_round'] = 501
model_index += 1
model_dict = {}
model_dict['model_type'] = 'Xgboost'
model_dict['model_params'] = tmp_params
models[model_index] = model_dict
'''


'''
## RandomForest models
rf_params = {'random_state' : 9999, 'n_estimators' : 2000, 'max_depth' : 7, 'criterion' : 'gini', 'n_jobs' : -1}

model_index += 1
model_dict = {}
model_dict['model_type'] = 'RandomForest'
model_dict['model_params'] = rf_params
models[model_index] = model_dict
'''

'''
rf_const_params = rf_params 
list_param_dict = {}
list_param_dict['n_estimators'] = [500, 1000, 2000, 3000]
list_param_dict['max_depth'] = [5, 7, 10, 15, 20, 25]
generate_models_params_dict(models, rf_const_params, list_param_dict, 'RandomForest', model_index)
'''

## ExtraTree models
et_params = {'random_state' : 9999, 'n_estimators' : 200, 'max_depth' : 18, 'criterion' : 'gini', 'n_jobs' : -1}

'''
model_index += 1
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = et_params
models[model_index] = model_dict
'''

#'''
et_const_params = et_params 
list_param_dict = {}
list_param_dict['n_estimators'] = [200, 500, 1000, 3000, 5000]
list_param_dict['max_depth'] = [3, 5, 7, 10, 15, 20, 25]
generate_models_params_dict(models, et_const_params, list_param_dict, 'ExtraTree', model_index)
#'''


with open(os.path.join(project_path, models_yaml_file), 'w') as yml_stream:
    yaml.dump(models, yml_stream, default_flow_style=False)

