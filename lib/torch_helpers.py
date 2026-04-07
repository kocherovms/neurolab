import torch.optim

class LrSchedulerWrapper:
    def __init__(self, optimizer, hp_learn_rate_params):
        if hp_learn_rate_params.plateau is None:
            self.scheduler = None
            self.step = self.step_dummy
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **hp_learn_rate_params.plateau._asdict())
            self.step = self.step_plateau

    def step_dummy(self, loss):
        pass
    
    def step_plateau(self, loss):
        self.scheduler.step(loss)

class ModelModeContextManager:
    def __init__(self, model, target_mode):
        self.training = model.training
        self.model = model

        match target_mode:
            case 'eval': self.model.eval()
            case 'train': self.model.train()
            case _: assert False, f'Unsupported {target_mode=}'

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.training:
            self.model.train()
        else:
            self.model.eval()
            
        return False

def eval_guard(model):
    return ModelModeContextManager(model, 'eval')

def train_guard(model):
    return ModelModeContextManager(model, 'train')
    