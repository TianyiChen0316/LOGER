class HashCache:
    @classmethod
    def hash(cls, value):
        return hash(value)

    def __init__(self):
        self.__cache = {}

    def put(self, key, value, hash=None):
        if hash is None:
            hash = self.hash(key)
        self.__cache[hash] = (key, value)

    def update(self, hash, value):
        origin = self.__cache.get(hash)
        if origin is not None:
            self.__cache[hash] = (origin[0], value)

    def get(self, hash, default=None):
        res = self.__cache.get(hash, None)
        if res is None:
            return default
        return res[1]

    def get_key(self, hash, default=None):
        res = self.__cache.get(hash, None)
        if res is None:
            return default
        return res[0]

    def __setitem__(self, key, value):
        self.put(key, value)

    def __getitem__(self, item):
        return self.get(item)

    def __len__(self):
        return len(self.__cache)

    def __bool__(self):
        return len(self.__cache) == 0

    def dump(self, copy=False):
        if copy:
            cache = self.__cache.copy()
        else:
            cache = self.__cache
        return cache

    def load(self, state_dict):
        self.__cache = state_dict
