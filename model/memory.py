import random
import typing

class Memory:
    def __init__(self, size=128):
        self.memory = []
        self.size = size
        self.__pointer = 0

    def load_state_dict(self, state_dict, load_config=False):
        if isinstance(state_dict, Memory):
            return self.load_state_dict(state_dict.state_dict(), load_config)
        if 'memory' in state_dict:
            self.memory = state_dict['memory']
        if load_config:
            if 'pointer' in state_dict:
                self.__pointer = state_dict['pointer']
            if 'size' in state_dict:
                self.size = state_dict['size']
        else:
            if 'size' in state_dict:
                diff = self.size - state_dict['size']
                if diff > 0:
                    self.__pointer = 0
                else:
                    if diff < 0:
                        self.memory = self.memory[:diff]
                    if 'pointer' in state_dict:
                        self.__pointer = state_dict['pointer']
            elif 'pointer' in state_dict:
                self.__pointer = state_dict['pointer']

    def state_dict(self):
        return {
            'memory': self.memory,
            'size': self.size,
            'pointer': self.__pointer,
        }

    def push(self, value):
        if len(self.memory) < self.size:
            self.memory.append(value)
        else:
            if self.__pointer >= len(self.memory):
                self.__pointer = 0
            self.memory[self.__pointer] = value
            self.__pointer += 1

    def sample(self, size):
        return random.sample(self.memory, k=min(size, len(self.memory)))

    def clear(self):
        self.memory = []
        self.__pointer = 0

class BestCache:
    def __init__(self, cmp : typing.Callable[[typing.Any, typing.Any], bool] = None):
        if cmp is None:
            cmp = lambda x, y: x < y
        self.cmp = cmp
        self.cache = {}

    def __setitem__(self, key, value):
        assert value is not None
        prev_value = self.cache.get(key, None)
        if prev_value is not None:
            if self.cmp(value, prev_value):
                self.cache[key] = value
        else:
            self.cache[key] = value

    def __getitem__(self, key):
        return self.cache[key]

    def __contains__(self, item):
        return item in self.cache

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def clear(self):
        self.cache.clear()
