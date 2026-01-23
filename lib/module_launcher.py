import sys
import os
import time
from dataclasses import dataclass

def if_verbose(verbosity, verbosity_threshold, func):
    if verbosity < verbosity_threshold:
        return

    func()

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
class LaunchParameters:
    module_fname: str
    func_name: str
    func_args: list
    verbosity: int

def launch(lp):
    module_dir_name = os.path.dirname(lp.module_fname)
    module_name = os.path.splitext(os.path.basename(lp.module_fname))[0]
    if_verbose(lp.verbosity, 3, lambda: print(f'{lp.module_fname=}, {module_dir_name=}, {module_name=}, {lp.func_name=}'))
    if_verbose(lp.verbosity, 2, lambda: print(f'Running {module_name}.{lp.func_name}{lp.func_args}'))
    sys.stdout.flush()

    sys.path.append(module_dir_name)
    module = importstr(*module_name.rsplit('.', 1)) 
    func = getattr(module, lp.func_name)
    rv = func(lp.func_args)

    if_verbose(lp.verbosity, 1, lambda: print(f'Finished {module_name}.{lp.func_name}{lp.func_args}, rv={rv}'))
    sys.stdout.flush()
    return rv
    
    