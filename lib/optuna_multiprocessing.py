import sys
import os
import time
from dataclasses import dataclass

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from logging_utils import *
from autoincrement import Autoincrement
import model_registry
import launchit

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

def get_objective(module_fname):
    def objective(trial):
        module_dir_name = os.path.dirname(module_fname)
        module_name = os.path.splitext(os.path.basename(module_fname))[0]
        Logging.trace(f'{module_fname=}, {module_dir_name=}, {module_name=}')
    
        sys.path.append(module_dir_name)
        ENVELOPE.trial = trial
        importstr(*module_name.rsplit('.', 1))
        result = ENVELOPE.result
    
        Logging.info(f'Finished {module_name}, result={result}')
        return result

    return objective

@dataclass
class RunOptimizationParameters:
    app_name: str
    is_stdout_enabled: bool
    notebook_fname: str
    notebook_name: str
    model_group_uri: str
    model_name: str
    expandvars: dict
    collect_inds: list
    run_path: str    
    study_name: str
    study_fname: str

# Launched from under the spawned process
def run_optimization(params):
    Logging.get().app_name = params.app_name
    Logging.get().enable('stdout', params.is_stdout_enabled)
    
    model_version = int(Autoincrement.get(f'{params.model_group_uri}.{params.model_name}'))
    assert model_version > 0, model_version
    model_registry_obj = model_registry.ModelRegistry(params.model_group_uri)
    model_registry_obj.register_model(params.model_name, model_version)
    Logging.info(f'Model instance registered, version={model_version}')

    params.expandvars['MODEL_VERSION'] = model_version
    module_fname = launchit.launchit(
        params.notebook_fname, 
        launch_serial=int(model_version),
        expandvars=params.expandvars, 
        make_py_file=True, 
        dir_name=params.run_path,
        collect_inds=params.collect_inds)
    Logging.info(f'Created "{module_fname}"')

    study = optuna.create_study(
        study_name=params.study_name,
        direction='maximize',
        storage=JournalStorage(JournalFileBackend(file_path=params.study_fname)),
        load_if_exists=True,
    )
    study.optimize(get_objective(module_fname), n_trials=1)