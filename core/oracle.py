import cx_Oracle as oracle
import pickle
import os
import time
import re

from lib.timer import timer as Timer
from . import config

class _oracle_db:
    auto_save_interval = 400

    @property
    def statement_timeout(self):
        return self.__statement_timeout

    @statement_timeout.setter
    def statement_timeout(self, value):
        self.__statement_timeout = value
        if self.__db is not None:
            self.__db.call_timeout = self.__statement_timeout
            self.__cur = self.__db.cursor()
            self.__cur.execute('select null from dual')
            self.__cur.fetchall()

    def __init__(self):
        self.__db = None
        self.__boundary_cache = {}
        self.__selectivity_cache = {}
        self.__latency_cache = {}
        self.__table_count_cache = {}
        self.__cost_cache = {}
        self.__plan_latency_cache = {}
        self.name = None
        self.__auto_save_count = 0
        self.__statement_timeout = 1000000

        self.__executed = {}

    def __auto_save(self):
        if not self.use_cache:
            return
        self.__auto_save_count += 1
        if self.auto_save_interval > 0 and self.__auto_save_count >= self.auto_save_interval:
            self.__auto_save_count = 0
            self.__cache_backup()

    def __cache_backup(self):
        if self.__db is None:
            return
        with open(f'.{self.name}.oracle_cache.pkl', 'wb') as f:
            pickle.dump(self.__boundary_cache, f)
            pickle.dump(self.__selectivity_cache, f)
            pickle.dump(self.__latency_cache, f)
            pickle.dump(self.__table_count_cache, f)
            pickle.dump(self.__cost_cache, f)
            pickle.dump(self.__plan_latency_cache, f)

    def __cache_load(self):
        if self.__db is None:
            return
        filename = f'.{self.name}.oracle_cache.pkl'
        if not os.path.isfile(filename):
            return
        with open(filename, 'rb') as f:
            self.__boundary_cache.update(pickle.load(f))
            self.__selectivity_cache.update(pickle.load(f))
            self.__latency_cache.update(pickle.load(f))
            self.__table_count_cache.update(pickle.load(f))
            self.__cost_cache.update(pickle.load(f))
            self.__plan_latency_cache.update(pickle.load(f))

    def __connect(self):
        self.__executed.clear()
        args, kwargs = self.__connection_args
        self.__db = oracle.connect(*args, **kwargs)
        if self.__statement_timeout is not None:
            self.__db.call_timeout = self.__statement_timeout
        self.__cur = self.__db.cursor()
        self.__cur.execute('alter system set statistics_level = all')
        self.__cur.execute('select null from dual')
        self.__cur.fetchall()

    def __execute(self, *args, **kwargs):
        try:
            self.__cur.execute(*args, **kwargs)
        except oracle.DatabaseError as e:
            s = str(e)
            if 'DPI-1080:' in s or 'DPI-1010:' in s:
                self.__connect()
            raise e

    _Re_plan_row = re.compile(r'^\|\*? *[0-9]+ *\|( *)([A-Za-z0-9_][A-Za-z0-9_ ]*)\|([A-Za-z0-9_ ]+)\|')

    def plan_left_deep(self, sql):
        self.__execute(f"explain plan for {sql}")
        self.__execute(f"select * from table(dbms_xplan.display)")
        res = self.__cur.fetchall()
        res_to_sort = []
        for index, (row, *_) in enumerate(res[5:]):
            m = re.search(self._Re_plan_row, row)
            if not m:
                break
            blank, method, table = m.groups()
            table = table.strip()
            level = len(blank)
            if method.strip().lower().startswith('table access'):
                res_to_sort.append((-level, index, table))
        res_to_sort = sorted(res_to_sort)
        _res = []
        for _, _, table in res_to_sort:
            _res.append(table)
        return _res

    def setup(self, *args, **kwargs):
        assert 'dbname' in kwargs
        if 'cache' in kwargs:
            self.use_cache = bool(kwargs['cache'])
            kwargs.pop('cache')
        else:
            self.use_cache = True

        self.name = kwargs.pop('dbname')
        self.__connection_args = (args, kwargs)
        self.__connect()
        self.config = config.Config()

        self.__pre_check()
        if self.use_cache:
            self.__cache_load()

    def __pre_check(self):
        pass

    def __pre_settings(self):
        pass

    def boundary(self, table, column):
        assert self.__db is not None
        query = (table, column)
        if query in self.__boundary_cache:
            return self.__boundary_cache[query]
        table_name = table.split(' ')[-1]
        self.__execute(f"select max({table_name}.{column}) from {table};")
        max_ = self.__cur.fetchall()[0][0]
        self.__execute(f"select min({table_name}.{column}) from {table};")
        min_ = self.__cur.fetchall()[0][0]
        res = (max_, min_)
        self.__boundary_cache[query] = res
        self.__auto_save()
        return res

    def table_size(self, table):
        assert self.__db is not None
        if table in self.__table_count_cache:
            return self.__table_count_cache[table]
        self.__execute(f'select count(*) from {table};')
        total_rows = self.__cur.fetchall()[0][0]
        self.__table_count_cache[table] = total_rows
        self.__auto_save()
        return total_rows

    def selectivity(self, table, where, explain=False):
        assert self.__db is not None
        query = (table, where, explain)
        if query in self.__selectivity_cache:
            return self.__selectivity_cache[query]
        total_rows = self.table_size(table)
        if explain:
            self.__execute(f"explain plan set statement_id = 'current' for select * from {table} where {where}")
            self.__execute(f"select cardinality from plan_table where statement_id = 'current'")
            select_rows = self.__cur.fetchall()[0][0]
            #select_rows = int(re.search(r'rows=([0-9]+)', select_rows).group(1))
        else:
            self.__execute(f'select count(*) from {table} where {where}')
            select_rows = self.__cur.fetchall()[0][0]
        res = select_rows / total_rows
        self.__selectivity_cache[query] = res
        self.__auto_save()
        return res

    def first_element(self, sql):
        assert self.__db is not None
        self.__pre_settings()
        self.__execute(sql)
        res = self.__cur.fetchall()
        return res[0][0]

    def cost(self, sql, cache=True):
        assert self.__db is not None
        if cache and sql in self.__cost_cache:
            return self.__cost_cache[sql]

        self.__execute(f"explain plan set statement_id = 'current' for {sql}")
        cost = self.first_element(f"select cost from plan_table where statement_id = 'current'")
        #cost = float(res.split("cost=")[1].split("..")[1].split(" ")[0])
        self.__db.commit()
        self.__cost_cache[sql] = cost
        self.__auto_save()
        return cost

    def plan_time(self, sql):
        assert self.__db is not None
        now = time.time()
        self.cost(sql)
        res = time.time() - now
        return res

    def plan(self, sql, geqo=False):
        raise NotImplementedError()

    def plan_latency(self, sql, cache=True):
        raise NotImplementedError()

    def table_rows(self, table_name, filter=None, schema_name=None, time_limit=None):
        assert self.__db is not None
        raise NotImplementedError()

    def result(self, sql):
        assert self.__db is not None
        #return iterator_utils.cursor_iter(self.__db, sql)
        self.__execute(sql)
        return self.__cur.fetchall()

    def table_columns(self, table_name, schema_name=None):
        assert self.__db is not None
        raise NotImplementedError()

    @classmethod
    def _sql_preprocess(cls, sql):
        sql = re.sub(r';\s*$', '', str(sql))
        sql = re.sub(r'[\r\a\v\b\f\t]', '', sql)
        _sql_lines = sql.split('\n')
        sql_lines = []
        for line in _sql_lines:
            if re.search(r'^ *--', line):
                # annotation line
                pass
            else:
                sql_lines.append(line)
        sql = ' '.join(sql_lines)
        return sql

    def __latency(self, sql, cache=True, detail=False):
        with Timer() as timer:
            self.__execute(sql)

        res = self.__cur.fetchall()
        self.__executed[sql] = res

        cost = timer.time * 1000
        if detail:
            return cost, res
        return cost

    def latency(self, sql, origin=None, cache=True):
        assert self.__db is not None
        sql = self._sql_preprocess(sql)
        timeout_limit = self.__timeout_limit(sql)
        self.__pre_settings()

        if origin is None:
            latency = None
            try:
                latency = self.__latency(sql, cache=cache)
            except Exception as e:
                print(f'{e.__class__.__name__}:', e, f'"""{sql}"""')
                self.__db.commit()
            if latency is None:
                latency = timeout_limit
            self.__latency_cache[sql] = latency
            self.__auto_save()
            return latency

        cost = self.cost(sql)
        cost_origin = self.cost(origin)

        latency = None
        res = None
        try:
            latency, res = self.__latency(sql, cache=cache, detail=True)
        except:
            self.__db.commit()
        if latency is None:
            latency = min(cost / cost_origin * self.latency(origin, cache=cache), timeout_limit)
        else:
            origin_res = self.__executed.get(str(origin), None)
            if origin_res is not None:
                assert origin_res == res, f'Result different from original SQL: \n\n{res}\n\n{origin_res}'
        self.__latency_cache[latency] = latency
        self.__auto_save()
        return latency

    def __timeout_limit(self, sql):
        assert self.__db is not None
        if sql in self.__latency_cache:
            return self.__latency_cache[sql] * 4 + self.plan_time(sql)
        return 1000000

    @property
    def schema(self):
        return self.__schema

oracle_database = _oracle_db()