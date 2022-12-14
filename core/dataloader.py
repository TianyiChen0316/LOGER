import os
from lib import filepath as fp
from tqdm import tqdm
from core.sql import Sql
import torch
from lib.timer import timer

def _load(directory, verbose=False):
    L = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            L.append(os.path.join(root, file))
    res = []
    if verbose:
        L = tqdm(L, desc='Loading files')
    for file in L:
        with open(file, 'r') as f:
            data = ' '.join(f.readlines())
            res.append((data, file))
    return res

def load(config, directory, device=torch.device('cpu'), verbose=False, detail=False):
    _timer = timer()
    _pth = fp.path_split(directory)
    _dir = os.sep.join(_pth[:-1])
    if detail:
        cache_file = f'{_dir}{os.sep}.{_pth[-1]}.detail.pkl'
    else:
        cache_file = f'{_dir}{os.sep}.{_pth[-1]}.pkl'
    if os.path.isfile(cache_file):
        return torch.load(cache_file, map_location=device)
    res = []
    _detail = []
    gen = _load(directory, verbose=verbose)
    if verbose:
        gen = tqdm(gen, desc='Parsing')
    for sql, filename in gen:
        fname = fp.path_split(filename)[-1]
        with _timer:
            sql = Sql(sql, config.feature_length, filename=fname)
        _detail.append((_timer.time))
        sql.to(device)
        res.append(sql)
    if detail:
        torch.save((res, _detail), cache_file)
        return res, _detail
    torch.save(res, cache_file)
    return res
