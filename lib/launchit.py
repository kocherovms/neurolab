import os
import sys
import json
import re
import string
from collections import namedtuple
from enum import IntEnum, auto 

from logging_utils import *

class Command(IntEnum):
    COLLECT = auto()
    COLLECT_1 = auto()
    COLLECT_2 = auto()
    COLLECT_3 = auto()
    COLLECTED = auto()
    DISABLE = auto()
    DISABLE_1 = auto()
    DISABLE_2 = auto()
    DISABLE_3 = auto()

ExecGraphEntry = namedtuple('ExecGraphEntry', 'command cell_ind source_line_ind is_oneliner stop_source_line_ind')

def launchit(fname, launch_serial=0, expandvars={}, make_py_file=False, dir_name='', max_serials_count=1_000, collect_inds=None, disable_inds=None):
    fname_dir = os.path.dirname(fname) if not dir_name else dir_name
    fname_name = os.path.splitext(os.path.basename(fname))[0]
    fname_ext = os.path.splitext(fname)[1] if not make_py_file else '.py'
    
    new_fname = ''
    serials_range = range(1, max_serials_count + 1) if launch_serial == 0 else [launch_serial]

    for i in serials_range:
        new_fname = os.path.join(fname_dir, f'{fname_name}-launch{i}{fname_ext}')
    
        if not os.path.exists(new_fname):
            break
    else:
        raise Exception(f'Failed to generate new launch file name: all variants are taken')

    Logging.debug(f'Creating {new_fname}')
    
    with open(fname, 'r') as f:
        nb = json.load(f)
        exec_graph = []
        collected_source_lines = []
    
        for cell_ind, cell in enumerate(nb['cells']):
            for source_line_ind, source_line in enumerate(cell['source']):
                source_line = source_line.strip()
                m = re.match(r'^(.*)#\s*@launchit\.(\w+)\s*$', source_line)
                
                if m:
                    Logging.trace(f'Cell {cell_ind}, launchit stanza: "{source_line}"')
                    before = m.group(1)
                    command = m.group(2)
                    ege = ExecGraphEntry(command=None, cell_ind=cell_ind, source_line_ind=source_line_ind, is_oneliner=re.match(r'[^\s]+', before), stop_source_line_ind=-1)
    
                    match command:
                        case 'disable':
                            ege = ege._replace(command=Command.DISABLE)
                        case 'disable_1':
                            ege = ege._replace(command=Command.DISABLE_1)
                        case 'disable_2':
                            ege = ege._replace(command=Command.DISABLE_2)
                        case 'disable_3':
                            ege = ege._replace(command=Command.DISABLE_3)
                        case 'collect':
                            ege = ege._replace(command=Command.COLLECT)
                        case 'collect_1':
                            ege = ege._replace(command=Command.COLLECT_1)
                        case 'collect_2':
                            ege = ege._replace(command=Command.COLLECT_2)
                        case 'collect_3':
                            ege = ege._replace(command=Command.COLLECT_3)
                        case 'collected':
                            assert not ege.is_oneliner, '@launchit.collected cannot be oneliner'
                            ege = ege._replace(command=Command.COLLECTED)
                        case 'stop':
                            assert not ege.is_oneliner, '@launchit.stop cannot be oneliner'
                            # look behind and patch stop_source_line_ind
                            for lb_ege_ind in range(len(exec_graph) - 1, -1, -1):
                                lb_ege = exec_graph[lb_ege_ind]
                                
                                if lb_ege.cell_ind != cell_ind:
                                    raise Exception(f'Cell {cell_ind}, line {source_line_ind}, @launchit.stop has no preceeding command')
                                elif lb_ege.stop_source_line_ind == -1:
                                    exec_graph[lb_ege_ind] = lb_ege._replace(stop_source_line_ind=source_line_ind)
                                    Logging.trace(f'Cell {cell_ind}, command {lb_ege.command.name} at line {lb_ege.source_line_ind} will stop at line {source_line_ind}')
                                    break
                        case _:
                            Logging.warn(f'WARNING! Cell {cell_ind} contains unrecognized launchit command: "{command}"')
    
                    if not ege.command is None:
                        exec_graph.append(ege)
    
        for ege in sorted(exec_graph):
            cell = nb['cells'][ege.cell_ind]
            stop_source_line_ind = len(cell['source']) if ege.stop_source_line_ind == -1 else ege.stop_source_line_ind
            
            match ege.command:
                case Command.COLLECT | Command.COLLECT_1 | Command.COLLECT_2 | Command.COLLECT_3:
                    do_collect = collect_inds is None

                    if not do_collect:
                        do_collect = ege.command == Command.COLLECT

                    if not do_collect:
                        collect_ind = {Command.COLLECT_1: 1, Command.COLLECT_2: 2, Command.COLLECT_3: 3}[ege.command]
                        do_collect = collect_ind in collect_inds

                    if not do_collect:
                        if ege.is_oneliner:
                            Logging.trace(f'Cell {ege.cell_ind}, skip collecting source line {ege.source_line_ind}')
                        else:
                            assert ege.source_line_ind + 1 < stop_source_line_ind
                            Logging.trace(f'Cell {ege.cell_ind}, skip collecting source lines from {ege.source_line_ind + 1} to {stop_source_line_ind}')
                    else:
                        if ege.is_oneliner:
                            Logging.trace(f'Cell {ege.cell_ind}, collecting source line {ege.source_line_ind}')
                            source_line = cell['source'][ege.source_line_ind]
                            collected_source_lines.append(source_line)
                        else:
                            assert ege.source_line_ind + 1 < stop_source_line_ind
                            Logging.trace(f'Cell {ege.cell_ind}, collecting source lines from {ege.source_line_ind + 1} to {stop_source_line_ind}')
        
                            if collected_source_lines:
                                collected_source_lines.append('\n')
            
                            for source_line_ind in range(ege.source_line_ind + 1, stop_source_line_ind):
                                source_line = cell['source'][source_line_ind]
                                collected_source_lines.append(source_line)
                case Command.COLLECTED:
                    Logging.trace(f'Cell {ege.cell_ind}, putting {len(collected_source_lines)} collected source lines to {ege.source_line_ind}')
    
                    for ind, source_line in enumerate(collected_source_lines):
                        if ind > 0:
                            cell['source'].insert(ege.source_line_ind + ind, source_line)
                        else:
                            cell['source'][ege.source_line_ind + ind] = source_line
                case Command.DISABLE | Command.DISABLE_1 | Command.DISABLE_2 | Command.DISABLE_3:
                    def disable_source_line(s):
                        do_disable = disable_inds is None

                        if not do_disable:
                            do_disable = ege.command == Command.DISABLE

                        if not do_disable:
                            disable_ind = {Command.DISABLE_1: 1, Command.DISABLE_2: 2, Command.DISABLE_3: 3}[ege.command]
                            do_disable = disable_ind in disable_inds

                        if not do_disable:
                            return s
                            
                        if re.match(r'^\s*#', s):
                            return s # already disabled
                        else:
                            return '# ' + s
                    
                    if ege.is_oneliner:
                        Logging.trace(f'Cell {ege.cell_ind}, disabling source line {ege.source_line_ind}')
                        cell['source'][ege.source_line_ind] = disable_source_line(cell['source'][ege.source_line_ind])
                    else:
                        assert ege.source_line_ind + 1 < stop_source_line_ind
                        Logging.trace(f'Cell {ege.cell_ind}, disabling source lines from {ege.source_line_ind + 1} to {stop_source_line_ind}')
    
                        for source_line_ind in range(ege.source_line_ind + 1, stop_source_line_ind):
                            cell['source'][source_line_ind] = disable_source_line(cell['source'][source_line_ind])
                case _:
                    assert False, f'Failed to understand exec_graph_entry={ege}'

        expandvars['LAUNCHIT_FNAME'] = new_fname
        Logging.trace(f'{expandvars=}')
        
        for cell in nb['cells']:
            for source_line_ind, source_line in enumerate(cell['source']):
                t = string.Template(source_line)
                cell['source'][source_line_ind] = t.safe_substitute(expandvars)
    
    with open(new_fname, 'w') as f:
        if make_py_file:
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    f.writelines(cell['source'])
                    f.write('\n\n')
        else:
            json.dump(nb, f, indent=2) # .ipynb
    
    return new_fname