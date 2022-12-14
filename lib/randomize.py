import torch
import numpy as np
import random
from torch import random as torch_random
from torch import cuda

torch.backends.cudnn.deterministic = True

def seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def get_random_state():
    torch_state = torch_random.get_rng_state()
    cuda_state = cuda.get_rng_state_all()
    np_state = np.random.get_state()
    state = random.getstate()
    return state, np_state, torch_state, cuda_state

def set_random_state(state):
    state, np_state, torch_state, cuda_state = state
    random.setstate(state)
    np.random.set_state(np_state)
    torch_random.set_rng_state(torch_state.to('cpu'))
    cuda_state = [s.to('cpu') for s in cuda_state]
    cuda.set_rng_state_all(cuda_state)

