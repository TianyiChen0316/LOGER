import time as _time
import typing as _typing

class timer:
    def __init__(self, callback: _typing.Callable[[_typing.Any], _typing.Any] = None):
        self.__time = None
        self.callback = callback

    def __enter__(self):
        self.__t = _time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__time = _time.time() - self.__t
        if self.callback is not None:
            self.callback(self)

    def reset(self):
        self.__time = None

    @property
    def time(self):
        return self.__time
