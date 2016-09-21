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


## RandomForest models
rf_const_params = {'random_state' : 9999, 'n_estimators' : 3600, 'max_depth' : 5, 'criterion' : 'entropy', 'n_jobs' : -1}
list_param_dict = {}
list_param_dict['max_depth'] = range(3, 8, 1)
list_param_dict['n_estimators'] = [1000, 2000, 3000, 4000]
list_param_dict['criterion'] = ['gini', 'entropy']
generate_models_params_dict(models, rf_const_params, list_param_dict, 'RandomForest', model_index)

## ExtraTree models
rf_const_params = {'random_state' : 9999, 'n_estimators' : 3600, 'max_depth' : 5, 'criterion' : 'entropy', 'n_jobs' : -1}
list_param_dict = {}
list_param_dict['max_depth'] = range(3, 8, 1)
list_param_dict['n_estimators'] = [300, 500, 700, 900]
list_param_dict['criterion'] = ['gini', 'entropy']
generate_models_params_dict(models, rf_const_params, list_param_dict, 'ExtraTree', model_index)


with open(os.path.join(project_path, models_yaml_file), 'w') as yml_stream:
    yaml.dump(models, yml_stream, default_flow_style=False)

