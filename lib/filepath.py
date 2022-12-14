import os as _os

__all__ = ["path_split", "in_path", "goto_path", "position", "pwd", "files", "dirs"]

def pwd() -> str:
    """
    Returns path of current directory.
    :return: the path of current directory
    """
    return _os.path.abspath(_os.curdir)

def path_split(path: str) -> list:
    """
    To split the path into sequence of directories and file name.
    :param path: the path to split
    :return: list of splitted path
    """
    res = []
    former = 0
    for i in range(len(path)):
        if path[i] == _os.sep or path[i] == _os.altsep:
            d = path[former:i]
            if d:
                res.append(d)
            former = i + 1
    if former < i:
        res.append(path[former:i + 1])
    return res

def parent(path: str) -> str:
    """
    To get parent directory of a file from its path.
    :param path: path of the file
    :return: file extension
    """
    rf = path.rfind(_os.sep)
    if _os.altsep is not None:
        rf1 = path.rfind(_os.altsep)
        rf = max(rf, rf1)
    if rf >= 0:
        return path[:rf]
    return ""

def filename(path: str) -> str:
    """
    To get the name of a file from its path.
    :param path: path of the file
    :return: file name
    """
    res = path_split(path)
    if not res:
        return ""
    return res[-1]

def ext(path: str) -> str:
    """
    To get extension of a file from its path.
    :param path: path of the file
    :return: file extension
    """
    rf = path.rfind(_os.extsep)
    if rf >= 0:
        return path[rf + 1:]
    return ""

def goto_path(path: str, abs: bool=False) -> None:
    """
    To go into the specified path.
    :param path: the path to go
    :return: None
    """
    dirs = path_split(path)
    if abs:
        _os.chdir(_os.sep)
    for i in range(len(dirs)):
        d = dirs[i]
        if not _os.path.exists(d):
            _os.mkdir(d)
            _os.chdir(d + _os.sep)
        elif not _os.path.isdir(d):
            sep = _os.altsep if _os.altsep is not None else _os.sep
            raise Exception("%s is not a folder" % (sep.join(dirs[0:i + 1])))
        else:
            _os.chdir(d + _os.sep)

def in_path(path: str) -> callable:
    """
    Returns a decorator, which wraps a function and when
      the function is called, the path is automatically
      opened, and returns to current path when the function
      is terminated.
    :param path: the path to go
    :return: function wrapper
    """
    def in_path(func):
        def wrapper(*args, **kwargs):
            nonlocal path
            cd = pwd()
            goto_path(path)
            try:
                return func(*args, **kwargs)
            except Exception:
                raise
            finally:
                goto_path(cd, abs=True)
        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = func.__qualname__
        return wrapper
    return in_path

class position:
    """
    The class of path opener.
    The usage is similar to @in_path(path). using
      position(path) to return a position object,
      which can be used in 'with' sentence, 
    """
    def __init__(self, path):
        self.__path = path

    def __enter__(self):
        self.path = pwd()
        goto_path(self.__path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        goto_path(self.path, abs=True)

    def __repr__(self):
        return self.__path

    def __str__(self):
        return self.__path

def files(path: str=None, extension: str=None) -> list:
    """
    To get list of files (except directories) from the path.
    When extension is given, the function returns files with
      specified extension only.
    :param path: the path to scan
    :param extension: file extension
    :return: list of files
    """
    if path is None:
        path = '..'
    else:
        path = str(path)
    if not _os.path.isdir(path):
        return []
    fs = _os.listdir(path)
    res = []
    for filename in fs:
        if _os.path.isfile(path + _os.sep + filename):
            if extension is not None and ext(filename) != extension:
                continue
            res.append(filename)
    return res

def dirs(path: str=None) -> list:
    """
    To get list of directories from the path.
    :param path: the path to scan
    :return: list of directories
    """
    if path is None:
        path = '..'
    else:
        path = str(path)
    if not _os.path.isdir(path):
        return []
    fs = _os.listdir(path)
    res = []
    for filename in fs:
        if _os.path.isdir(path + _os.sep + filename):
            res.append(filename)
    return res

def ls(path: str=None) -> list:
    """
    To get list of files and directories from the path.
    :param path: the path to scan
    :return: list of files and directories
    """
    if path is None:
        path = '..'
    else:
        path = str(path)
    if not _os.path.isdir(path):
        return []
    fs = _os.listdir(path)
    res = []
    for filename in fs:
        res.append(filename)
    return res
