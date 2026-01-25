import sys
import os
import time
from dataclasses import dataclass

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from lib.logging_utils import if_verbose
import lib.model_registry
import lib.launchit

def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
            
    return module

@dataclass
class Envelope:
    trial: object = None
    result: object = None

ENVELOPE = Envelope()

def get_trial():
    return ENVELOPE.trial

def save_trial_result(result):
    ENVELOPE.result = result

def get_objective(module_fname, verbosity):
    def objective(trial):
        module_dir_name = os.path.dirname(module_fname)
        module_name = os.path.splitext(os.path.basename(module_fname))[0]
        if_verbose(verbosity, 3, lambda: print(f'{module_fname=}, {module_dir_name=}, {module_name=}'))
    
        sys.path.append(module_dir_name)
        ENVELOPE.trial = trial
        importstr(*module_name.rsplit('.', 1))
        result = ENVELOPE.result
    
        if_verbose(verbosity, 1, lambda: print(f'Finished {module_name}, result={result}'))
        return result

    return objective

@dataclass
class RunOptimizationParameters:
    notebook_fname: str
    notebook_name: str
    model_group_uri: str
    model_name: str
    expandvars: dict
    collect_inds: list
    run_path: str    
    verbosity: int = 0

# launched within a clean spawned process
# execution flow: [main_process: mp.Pool] -> [child_process: run_optimization -> objective -> run module (via import)]
# def run_optimization(params):
#     if_verbose(params.verbosity, 3, lambda: print(f'{params=}'))
#     study = optuna.create_study(
#         study_name=params.study_name,
#         direction=params.direction,
#         storage=JournalStorage(JournalFileBackend(file_path=params.journal_fname)),
#         load_if_exists=True,
#     )
#     study.optimize(get_objective(params.module_fname, params.verbosity), n_trials=1)

def run_optimization(params):
    if_verbose_ = lambda thres, func: if_verbose(params.verbosity, thres, func)
    if_verbose_(3, lambda: print(f'{params=}'))
    
    model_registry = lib.model_registry.ModelRegistry(params.model_group_uri)
    model_version = model_registry.register_model(params.model_name)
    assert model_version > 0, model_version
    if_verbose_(1, lambda: print(f'Model instance registered, version={model_version}'))

    params.expandvars['MODEL_VERSION'] = model_version
    module_fname = lib.launchit.launchit(
        params.notebook_fname, 
        launch_serial=int(model_version),
        expandvars=params.expandvars, 
        verbosity=params.verbosity, 
        make_py_file=True, 
        dir_name=params.run_path,
        collect_inds=params.collect_inds)
    if_verbose_(1, lambda: print(f'Created "{module_fname}"'))

    study = optuna.create_study(
        study_name=params.notebook_name,
        direction='maximize',
        storage=JournalStorage(JournalFileBackend(file_path=os.path.join(params.run_path, params.notebook_name + '.log'))),
        load_if_exists=True,
    )
    study.optimize(get_objective(module_fname, params.verbosity), n_trials=1)