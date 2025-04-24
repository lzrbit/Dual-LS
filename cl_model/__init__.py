import os
import importlib
# from utils.args_loading import root_dir

def get_all_models_name(args):
    return [model.split('.')[0] for model in os.listdir(args.root_dir + 'cl_model')
            if not model.find('__') > -1 and 'py' in model]


def get_all_models(args):
    names = {}
    for model in get_all_models_name(args):
        mod = importlib.import_module('cl_model.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '_')]
        names[model] = getattr(mod, class_name)
    return names

def get_model(names, args, backbone, loss):
    return names[args.model](backbone, loss, args)
