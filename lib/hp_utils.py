from collections import namedtuple
from dataclasses import dataclass
import lark

@dataclass
class LinearModelUnitParams: 
    size: int = None
    nonlinearity: str = None # any of: None, sigmoid, tanh, relu, ...
    dropout: float = None
    
@dataclass
class StateModelUnitParams: 
    module: str = None
    hidden_size: int = None
    num_layers: int = None

@dataclass
class LearnRateParams: 
    Plateau = namedtuple('Plateau', 'factor patience')
    
    learn_rate: float = None
    plateau: object = None

def get_lark_tree_value(tree, var_name, default_value=None):
    try:
        return next(tree.scan_values(lambda i: i.type == var_name)).value
    except StopIteration:
        return default_value

def hp_parse_model_units(units, expand_vars={}):
    grammar = '''
        spec: linear_spec | lstm_spec
    
        linear_spec: "Linear" "(" (dropout_spec "->")? linear_size_spec ("->" LINEAR_NONLINEARITY)? ")"
        linear_size_spec: expand_var_spec | LINEAR_SIZE
        LINEAR_SIZE: NUMBER
        LINEAR_NONLINEARITY: WORD
        
        dropout_spec: "dropout" "(" DROPOUT ")"
        DROPOUT: NUMBER
        
        lstm_spec: "LSTM" "(" LSTM_HIDDEN_SIZE ")" ("x" LSTM_NUM_LAYERS)?
        LSTM_HIDDEN_SIZE: NUMBER
        LSTM_NUM_LAYERS: NUMBER

        expand_var_spec: "$" EXPAND_VAR_NAME
        EXPAND_VAR_NAME: ("a".."z" | "0".."9" | "_")+
        
        %import common.WORD
        %import common.NUMBER
        %import common.WS
        %ignore WS
    '''
    parser = lark.Lark(grammar, start='spec')
    params_list = []

    for unit in units:
        tree = parser.parse(unit)
        gtv = lambda var_name, default_value='': get_lark_tree_value(tree, var_name, default_value)

        if spec_subtree := list(tree.find_data('linear_spec')):
            params = LinearModelUnitParams()
            linear_size_tree = gtv('LINEAR_SIZE')

            if t := list(spec_subtree[0].find_data('expand_var_spec')):
                expand_var_name = get_lark_tree_value(t[0], 'EXPAND_VAR_NAME')
                params.size = int(expand_vars[expand_var_name])
            else:
                params.size = int(gtv('LINEAR_SIZE'))
                
            params.nonlinearity = gtv('LINEAR_NONLINEARITY', None)
            params.dropout = float(gtv('DROPOUT')) if gtv('DROPOUT', False) else None
        elif list(tree.find_data('lstm_spec')):
            params = StateModelUnitParams()
            params.module = 'LSTM'
            params.hidden_size = int(gtv('LSTM_HIDDEN_SIZE'))
            params.num_layers = int(gtv('LSTM_NUM_LAYERS', 1))
        else:
            assert False, f'Unsupported unit spec="{unit}"'

        params_list.append(params)

    return params_list

def hp_parse_learn_rate(learn_rate):
    params = LearnRateParams()

    if isinstance(learn_rate, float):
        params.learn_rate = learn_rate
        return params
        
    grammar = '''
        spec: initial_lr_spec(","plateau_spec)?
    
        initial_lr_spec: INITIAL_LR
        INITIAL_LR: NUMBER
    
        plateau_spec: "plateau" "(" (|plateau_params_spec ("," plateau_params_spec)*) ")"
        plateau_params_spec: plateau_factor_spec | plateau_patience_spec
        plateau_factor_spec: "factor" "=" plateau_factor_value_spec
        plateau_factor_value_spec: PLATEAU_FACTOR_VALUE
        plateau_patience_spec: "patience" "=" plateau_patience_value_spec
        plateau_patience_value_spec: PLATEAU_PATIENCE_VALUE
        PLATEAU_FACTOR_VALUE: NUMBER
        PLATEAU_PATIENCE_VALUE: NUMBER
        
        %import common.NUMBER
        %import common.WS
        %ignore WS
    '''
    parser = lark.Lark(grammar, start='spec')
    tree = parser.parse(learn_rate)
    gtv = lambda var_name, default_value='': get_lark_tree_value(tree, var_name, default_value)
    params.learn_rate = float(gtv('INITIAL_LR'))
    
    if list(tree.find_data('plateau_spec')):
        params.plateau = LearnRateParams.Plateau(
            factor=float(gtv('PLATEAU_FACTOR_VALUE', 0.1)),
            patience=int(gtv('PLATEAU_PATIENCE_VALUE', 10)),
        )

    return params
