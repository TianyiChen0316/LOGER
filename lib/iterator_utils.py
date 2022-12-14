import os, sys, math
from collections import Sized
from ._postgres import Cursor

class SizedWrapper:
    def __init__(self, iter, size):
        self.size = size
        self.iter = iter

    def __getattr__(self, item, default=NotImplemented):
        if default is NotImplemented:
            return getattr(self.iter, item)
        return getattr(self.iter, item, default)

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.iter

    def __next__(self):
        return next(self.iter)


def fetcher(cursor : Cursor):
    res = cursor.fetchone()
    while res is not None:
        yield res
        res = cursor.fetchone()
    cursor.close()

def unpack(it):
    for i in it:
        for j in i:
            yield j

def cursor_to_data(cursor, filter_type=None):
    if filter_type is not None:
        return map(lambda x: tuple(filter(lambda y: isinstance(y, filter_type), x)), fetcher(cursor))
    return fetcher(cursor)

def _batchloader(data, batch_size):
    count = 0
    batch = []
    for i in data:
        batch.append(i)
        count = count + 1
        if count < batch_size:
            continue
        yield batch
        count = 0
        batch = []
    if batch:
        yield batch

def batchloader(data, batch_size):
    it = _batchloader(data, batch_size)
    if isinstance(data, Sized):
        return SizedWrapper(it, math.ceil(len(data) / batch_size))
    return it

def cursor_iter(database, select, filter_type=None):
    cursor = database.cursor()
    cursor.execute(select)
    data = cursor_to_data(cursor, filter_type=filter_type)
    return SizedWrapper(data, cursor.rowcount)

def cursor_iter_gen(database, select, filter_type=None):
    def cursor_iter_gen():
        return cursor_iter(database, select, filter_type=filter_type)
    return cursor_iter_gen

def file_iter(files, chunk_size=1000, encoding='utf8', whitespace=True):
    if isinstance(files, str):
        files = [files]
    for file in files:
        with open(file, 'r', encoding=encoding) as f:
            s = f.read(chunk_size)
            while len(s) != 0:
                if whitespace:
                    c = f.read(1)
                    while c != ' ' and len(c) != 0:
                        s += c
                        c = f.read(1)
                yield s
                s = f.read(chunk_size)

def file_iter_gen(files, chunk_size=1000, encoding='utf8', whitespace=False, verbose=False):
    if isinstance(files, str):
        files = [files]
    total = 0
    for file in files:
        total += math.ceil(os.path.getsize(file) / chunk_size)

    if verbose:
        print("%d chunks in total" % (total), file=sys.stderr)

    def file_iter_gen():
        it = file_iter(files, chunk_size=chunk_size, encoding=encoding, whitespace=whitespace)
        return SizedWrapper(it, total)
    return file_iter_gen

def file_sentence_iter(files, encoding='utf8', splitter=('.', '\n', ), whitespace=True):
    if isinstance(files, str):
        files = [files]
    for file in files:
        with open(file, 'r', encoding=encoding) as f:
            s = []
            c = f.read(1)
            while len(c) != 0:
                if not whitespace or not ((c == ' ' or c == '\n') and len(s) == 0):
                    s.append(c)
                if c in splitter:
                    yield ''.join(s)
                    s = []
                c = f.read(1)
            if s:
                yield ''.join(s)

def file_sentence_iter_gen(files, encoding='utf8', splitter=('.', '\n', ), whitespace=True):
    if isinstance(files, str):
        files = [files]
    def file_sentence_iter_gen():
        return file_sentence_iter(files, encoding=encoding, splitter=splitter, whitespace=whitespace)
    return file_sentence_iter_gen

def sized_map(func, iter):
    res = map(func, iter)
    if isinstance(iter, Sized):
        return SizedWrapper(res, len(iter))
    return res

def concat(iters):
    sz = 0
    for it in iters:
        if not isinstance(it, Sized):
            sz = -1
            break
        else:
            sz += len(it)
    def concat():
        for it in iters:
            for i in it:
                yield i
    if sz >= 0:
        return SizedWrapper(concat(), sz)
    return concat()

class StringStream:
    def __init__(self, *iters):
        self.__iters = list(reversed(iters))
        self.__buf = []
        self.__end = False
        self.__iter = self.__iters.pop() if self.__iters else None

    def read(self, n):
        res = []
        while n > 0 and not self.__end:
            nex = ''
            if self.__buf:
                nex = self.__buf.pop()
            elif self.__iter is not None:
                while self.__iter:
                    try:
                        nex = next(self.__iter)
                        break
                    except StopIteration:
                        self.__iter = self.__iters.pop() if self.__iters else None
            if self.__iter is None:
                self.__end = True
            n -= len(nex)
            if n < 0:
                nex, rest = nex[:-n], nex[-n:]
                self.__buf.append(rest)
                res.append(nex)
        return ''.join(res)

    def unread(self, s):
        self.__buf.append(s)
        self.__end = False

    def put(self, *iters):
        self.__iters.extend(iters)
        self.__end = False

    @property
    def end(self):
        return self.__end

