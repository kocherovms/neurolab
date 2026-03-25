import sys
import os
import time
from dataclasses import dataclass

import optuna
import optuna.samplers
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

def get_objective(module_fname, study_serial, study_name):
    def objective(trial):
        module_dir_name = os.path.dirname(module_fname)
        module_name = os.path.splitext(os.path.basename(module_fname))[0]
        Logging.trace(f'{module_fname=}, {module_dir_name=}, {module_name=}')
    
        sys.path.append(module_dir_name)
        ENVELOPE.trial = trial
        trial.set_user_attr('STUDY_SERIAL', study_serial)
        trial.set_user_attr('STUDY_NAME', study_name)

        with Logging.get().auto_prefix('OSTUDY', study_serial, 'OTRIAL', trial.number):
            importstr(*module_name.rsplit('.', 1))
            
        result = ENVELOPE.result
    
        Logging.info(f'Finished {module_name}, result={result}')
        return result

    return objective

@dataclass
class RunOptimizationTask:
    app_name: str
    is_stdout_enabled: bool
    notebook_fname: str
    notebook_name: str
    model_group_uri: str
    model_name: str
    expandvars: dict
    collect_inds: list
    disable_inds: list
    run_path: str  
    study_serial: int
    study_name: str
    study_fname: str
    optimize_directions: list
    grid_search_space: dict

# Launched from under the spawned process
def run_optimization(task):
    Logging.get().app_name = task.app_name
    Logging.get().enable('stdout', task.is_stdout_enabled)
    
    model_version = int(Autoincrement.get(f'{task.model_group_uri}.{task.model_name}'))
    assert model_version > 0, model_version
    model_registry_obj = model_registry.ModelRegistry(task.model_group_uri)
    model_registry_obj.register_model(task.model_name, model_version)
    Logging.info(f'Model instance registered, version={model_version}')

    with Logging.get().auto_prefix('MVER', model_version):
        task.expandvars['MODEL_VERSION'] = model_version
        module_fname = launchit.launchit(
            task.notebook_fname, 
            launch_serial=int(model_version),
            expandvars=task.expandvars, 
            make_py_file=True, 
            dir_name=task.run_path,
            collect_inds=task.collect_inds,
            disable_inds=task.disable_inds,
        )
        Logging.info(f'Created "{module_fname}"')
    
        study = optuna.create_study(
            study_name=task.study_name,
            directions=task.optimize_directions,
            storage=JournalStorage(JournalFileBackend(file_path=task.study_fname)),
            load_if_exists=True,
            sampler=optuna.samplers.GridSampler(task.grid_search_space) if task.grid_search_space is not None else None,
        )
        study.optimize(get_objective(module_fname, task.study_serial, task.study_name), n_trials=1)