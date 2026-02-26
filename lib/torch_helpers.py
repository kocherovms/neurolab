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