import os
import sys
import json
import re
import string
from collections import namedtuple
from enum import IntEnum, auto 

class Command(IntEnum):
    COLLECT = auto()
    COLLECTED = auto()
    DISABLE = auto()    

ExecGraphEntry = namedtuple('ExecGraphEntry', 'command cell_ind source_line_ind is_oneliner stop_source_line_ind')

def launchit(fname, launch_serial=0, expandvars={}):
    fname_dir = os.path.dirname(fname)
    fname_name = os.path.splitext(os.path.basename(fname))[0]
    fname_ext = os.path.splitext(fname)[1]
    
    new_fname = ''
    serials_range = range(1, 1_000 + 1) if launch_serial == 0 else [launch_serial]

    for i in serials_range:
        new_fname = os.path.join(fname_dir, f'{fname_name}-launch{i}{fname_ext}')
    
        if not os.path.exists(new_fname):
            break
    else:
        raise Exception(f'Failed to generate new launch file name: all variants are taken')
    
    print(f'Creating {new_fname}')
    
    with open(fname, 'r') as f:
        nb = json.load(f)
        exec_graph = []
        collected_source_lines = []
    
        for cell_ind, cell in enumerate(nb['cells']):
            for source_line_ind, source_line in enumerate(cell['source']):
                source_line = source_line.strip()
                m = re.match(r'^(.*)#\s*@launchit\.(\w+)\s*$', source_line)
                
                if m:
                    print(f'Cell {cell_ind}, launchit stanza: "{source_line}"')
                    before = m.group(1)
                    command = m.group(2)
                    ege = ExecGraphEntry(command=None, cell_ind=cell_ind, source_line_ind=source_line_ind, is_oneliner=re.match(r'[^\s]+', before), stop_source_line_ind=-1)
    
                    match command:
                        case 'disable':
                            ege = ege._replace(command=Command.DISABLE)
                        case 'collect':
                            ege = ege._replace(command=Command.COLLECT)
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
                                    print(f'Cell {cell_ind}, command {lb_ege.command.name} at line {lb_ege.source_line_ind} will stop at line {source_line_ind}')
                                    break
                        case _:
                            print(f'WARNING! Cell {cell_ind} contains unrecognized launchit command: "{command}"')
    
                    if not ege.command is None:
                        exec_graph.append(ege)
    
        for ege in sorted(exec_graph):
            cell = nb['cells'][ege.cell_ind]
            stop_source_line_ind = len(cell['source']) if ege.stop_source_line_ind == -1 else ege.stop_source_line_ind
            
            match ege.command:
                case Command.COLLECT:
                    if ege.is_oneliner:
                        print(f'Cell {ege.cell_ind}, collecting source line {ege.source_line_ind}')
                        source_line = cell['source'][ege.source_line_ind]
                        collected_source_lines.append(source_line)
                    else:
                        assert ege.source_line_ind + 1 < stop_source_line_ind
                        print(f'Cell {ege.cell_ind}, collecting source lines from {ege.source_line_ind + 1} to {stop_source_line_ind}')
    
                        if collected_source_lines:
                            collected_source_lines.append('\n')
        
                        for source_line_ind in range(ege.source_line_ind + 1, stop_source_line_ind):
                            source_line = cell['source'][source_line_ind]
                            collected_source_lines.append(source_line)
                case Command.COLLECTED:
                    print(f'Cell {ege.cell_ind}, putting {len(collected_source_lines)} collected source lines to {ege.source_line_ind}')
    
                    for ind, source_line in enumerate(collected_source_lines):
                        if ind > 0:
                            cell['source'].insert(ege.source_line_ind + ind, source_line)
                        else:
                            cell['source'][ege.source_line_ind + ind] = source_line
                case Command.DISABLE:
                    def disable_source_line(s):
                        if re.match(r'^\s*#', s):
                            return s # already disabled
                        else:
                            return '# ' + s
                    
                    if ege.is_oneliner:
                        print(f'Cell {ege.cell_ind}, disabling source line {ege.source_line_ind}')
                        cell['source'][ege.source_line_ind] = disable_source_line(cell['source'][ege.source_line_ind])
                    else:
                        assert ege.source_line_ind + 1 < stop_source_line_ind
                        print(f'Cell {ege.cell_ind}, disabling source lines from {ege.source_line_ind + 1} to {stop_source_line_ind}')
    
                        for source_line_ind in range(ege.source_line_ind + 1, stop_source_line_ind):
                            cell['source'][source_line_ind] = disable_source_line(cell['source'][source_line_ind])
                case _:
                    assert False, f'Failed to understand exec_graph_entry={ege}'

        expandvars['LAUNCHIT_FNAME'] = new_fname
        
        for cell in nb['cells']:
            for source_line_ind, source_line in enumerate(cell['source']):
                t = string.Template(source_line)
                cell['source'][source_line_ind] = t.safe_substitute(expandvars)
    
    with open(new_fname, 'w') as f:
        json.dump(nb, f, indent=2)
    
    print(f'Created "{new_fname}"')