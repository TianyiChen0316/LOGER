import sys

class Logger:
    def __init__(self, file=sys.stdout, buffering=None, stdout=False, stderr=False):
        if isinstance(file, str):
            args = {}
            if buffering is not None:
                args['buffering'] = buffering
            file = open(file, 'a', **args)
        self.file = file
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, *value, sep=' ', end='\n', flush=False):
        value = sep.join(map(str, value)) + end
        self.file.write(value)
        if self.stdout:
            print(value, file=sys.stdout, end='')
        if self.stderr:
            print(value, file=sys.stderr, end='')
        if flush:
            self.file.flush()
