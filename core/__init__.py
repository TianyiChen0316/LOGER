from .config import Config
from .sql import database, Sql, Baseline
from .plan import Plan
from .dataloader import *

import torch as _torch

class condition_grad(_torch.no_grad):
    def __init__(self, grad=False):
        super().__init__()
        self.grad = grad

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(self.grad)
