import random

class Explorer:
    def __init__(self):
        self.count = 0

    def reset(self):
        self.count = 0

    def step(self):
        self.count += 1

    def prob(self):
        pass

    def explore(self):
        return random.random() < self.prob

class HalfTimeExplorer(Explorer):
    def __init__(self, start, end, half_steps):
        super().__init__()
        self.start = start
        self.end = end
        self.half_steps = half_steps

    @property
    def prob(self):
        process = 0.5 ** (self.count / self.half_steps)
        return self.start + (self.end - self.start) * (1 - process)

class LinearExplorer(Explorer):
    def __init__(self, start, end, steps):
        super().__init__()
        self.start = start
        self.end = end
        self.steps = steps

    @property
    def prob(self):
        if self.count > self.steps:
            return self.end
        return self.start + (self.end - self.start) * self.count / self.steps
