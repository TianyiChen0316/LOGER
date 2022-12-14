from psqlparse import parse_dict
import torch
import numpy as np
import time
import re
import dgl
import pickle
import sys
from tqdm import tqdm
from lib import iterator_utils, postgres, filepath

from core import config


class _global:
    geqo_threshold = 2
    statement_timeout = 1000000
    auto_save_interval = 400

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

    @property
    def connection(self):
        return self.__db

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
        with open(f'.{self.name}.cache.pkl', 'wb') as f:
            pickle.dump(self.__boundary_cache, f)
            pickle.dump(self.__selectivity_cache, f)
            pickle.dump(self.__latency_cache, f)
            pickle.dump(self.__table_count_cache, f)
            pickle.dump(self.__cost_cache, f)
            pickle.dump(self.__plan_latency_cache, f)

    def __cache_load(self):
        if self.__db is None:
            return
        filename = f'.{self.name}.cache.pkl'
        if not os.path.isfile(filename):
            return
        with open(filename, 'rb') as f:
            self.__boundary_cache.update(pickle.load(f))
            self.__selectivity_cache.update(pickle.load(f))
            self.__latency_cache.update(pickle.load(f))
            self.__table_count_cache.update(pickle.load(f))
            self.__cost_cache.update(pickle.load(f))
            self.__plan_latency_cache.update(pickle.load(f))

    def assistant_setup(self, *args, **kwargs):
        assert 'dbname' in kwargs
        if 'cache' in kwargs:
            self.use_cache = bool(kwargs['cache'])
            kwargs.pop('cache')
        else:
            self.use_cache = True

        self.name = kwargs['dbname']
        self.config = config.Config()
        self.__schema = Schema(None, self)

        if self.use_cache:
            self.__cache_load()

    def setup(self, *args, **kwargs):
        assert 'dbname' in kwargs
        if 'cache' in kwargs:
            self.use_cache = bool(kwargs['cache'])
            kwargs.pop('cache')
        else:
            self.use_cache = True

        self.name = kwargs['dbname']
        self.__db = postgres.connect(*args, **kwargs)
        self.__cur = self.__db.cursor()
        self.config = config.Config()
        self.__schema = Schema(self.__db, self)

        self.__pre_check()
        if self.use_cache:
            self.__cache_load()

    def get_settings(self, key):
        assert self.__db is not None
        self.__cur.execute(f"show {key};")
        return self.__cur.fetchall()[0][0]

    def set_settings(self, key, value):
        assert self.__db is not None
        self.__cur.execute(f"set {key}=%s;", (value, ))

    def __getitem__(self, item):
        return self.get_settings(item)

    def __setitem__(self, key, value):
        self.set_settings(key, value)

    @property
    def enable_nestloop(self):
        return self.get_settings('enable_nestloop')

    @enable_nestloop.setter
    def enable_nestloop(self, value):
        self.set_settings('enable_nestloop', value)

    @property
    def enable_hashjoin(self):
        return self.get_settings('enable_hashjoin')

    @enable_hashjoin.setter
    def enable_hashjoin(self, value):
        self.set_settings('enable_hashjoin', value)

    @property
    def enable_mergejoin(self):
        return self.get_settings('enable_mergejoin')

    @enable_mergejoin.setter
    def enable_mergejoin(self, value):
        self.set_settings('enable_mergejoin', value)

    @property
    def enable_indexscan(self):
        return self.get_settings('enable_indexscan')

    @enable_indexscan.setter
    def enable_indexscan(self, value):
        self.set_settings('enable_indexscan', value)

    @property
    def enable_seqscan(self):
        return self.get_settings('enable_seqscan')

    @enable_seqscan.setter
    def enable_seqscan(self, value):
        self.set_settings('enable_seqscan', value)

    @property
    def enable_indexonlyscan(self):
        return self.get_settings('enable_indexonlyscan')

    @enable_indexonlyscan.setter
    def enable_indexonlyscan(self, value):
        self.set_settings('enable_indexonlyscan', value)

    @property
    def session_preload_libraries(self):
        assert self.__db is not None
        self.__cur.execute("show session_preload_libraries;")
        return list(map(lambda x: x.strip(), self.__cur.fetchall()[0][0].split(',')))

    @property
    def shared_preload_libraries(self):
        assert self.__db is not None
        self.__cur.execute("show shared_preload_libraries;")
        return list(map(lambda x: x.strip(), self.__cur.fetchall()[0][0].split(',')))

    def __pre_check(self):
        if self.config.use_hint:
            #self.__cur.execute("load 'pg_hint_plan';")
            assert 'pg_hint_plan' in self.session_preload_libraries + self.shared_preload_libraries, \
                "pg_hint_plan is not loaded"

    def __pre_settings(self):
        if self.config.use_hint:
            self.__cur.execute("set from_collapse_limit = 1;")
            self.__cur.execute("set join_collapse_limit = 1;")
            self.__cur.execute(f"set geqo_threshold = 1024;")
        else:
            self.__cur.execute("set from_collapse_limit = 1;")
            self.__cur.execute("set join_collapse_limit = 1;")
            self.__cur.execute(f"set geqo_threshold = {self.geqo_threshold};")
        self.__cur.execute("set max_parallel_workers = 1;")
        self.__cur.execute("set max_parallel_workers_per_gather = 1;")
        self.__cur.execute(f"SET statement_timeout = {self.statement_timeout};")

    def boundary(self, table, column):
        assert self.__db is not None
        query = (table, column)
        if query in self.__boundary_cache:
            return self.__boundary_cache[query]
        table_name = table.split(' ')[-1]
        self.__cur.execute(f"select max({table_name}.{column}) from {table};")
        max_ = self.__cur.fetchall()[0][0]
        self.__cur.execute(f"select min({table_name}.{column}) from {table};")
        min_ = self.__cur.fetchall()[0][0]
        res = (max_, min_)
        self.__boundary_cache[query] = res
        self.__auto_save()
        return res

    def table_size(self, table):
        assert self.__db is not None
        if table in self.__table_count_cache:
            return self.__table_count_cache[table]
        self.__cur.execute(f'select count(*) from {table};')
        total_rows = self.__cur.fetchall()[0][0]
        self.__table_count_cache[table] = total_rows
        self.__auto_save()
        return total_rows

    def selectivity(self, table, where, explain=False, detail=False):
        assert self.__db is not None
        query = (table, where, explain)
        if query in self.__selectivity_cache:
            if detail:
                return self.__selectivity_cache[query]
            return self.__selectivity_cache[query][0]
        total_rows = self.table_size(table)
        if explain:
            self.__cur.execute(f'explain select * from {table} where {where};')
            select_rows = self.__cur.fetchall()[0][0]
            select_rows = int(re.search(r'rows=([0-9]+)', select_rows).group(1))
        else:
            self.__cur.execute(f'select count(*) from {table} where {where};')
            select_rows = self.__cur.fetchall()[0][0]
        res = select_rows / total_rows
        res = (res, select_rows, total_rows)
        self.__selectivity_cache[query] = res
        self.__auto_save()
        if detail:
            return res
        return res[0]

    def first_element(self, sql):
        assert self.__db is not None
        self.__pre_settings()
        self.__cur.execute(sql)
        res = self.__cur.fetchall()
        return res[0][0]

    def cost(self, sql, cache=True):
        assert self.__db is not None
        if cache and sql in self.__cost_cache:
            return self.__cost_cache[sql]

        res = self.first_element(f'EXPLAIN {sql}')
        cost = float(res.split("cost=")[1].split("..")[1].split(" ")[0])
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
        self.__pre_settings()

        sql = sql + ";"
        if geqo:
            self.__cur.execute(f"set geqo_threshold = 2;")
        try:
            self.__cur.execute("EXPLAIN (COSTS, FORMAT JSON) " + sql)
            rows = self.__cur.fetchall()
        except:
            print(sql)
            raise
        finally:
            self.__cur.execute(f"set geqo_threshold = {self.geqo_threshold};")
        return rows

    def plan_latency(self, sql, cache=True):
        self.__pre_settings()
        sql = sql + ";"

        if cache:
            cached = self.__plan_latency_cache.get(sql, None)
            if cached is not None:
                return cached

        self.__cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        rows = self.__cur.fetchall()
        self.__plan_latency_cache[sql] = rows
        self.__auto_save()

        return rows

    def table_rows(self, table_name, filter=None, schema_name=None, time_limit=None):
        assert self.__db is not None
        if time_limit is None:
            return list(postgres.filter_iter_table(self.__db, table_name, schema_name=schema_name, filter=filter))
        self.__cur.execute(f"SET statement_timeout = {time_limit};")
        res = None
        try:
            res = list(postgres.filter_iter_table(self.__db, table_name, schema_name=schema_name, filter=filter))
        except:
            pass
        finally:
            self.__db.commit()
            self.__cur.execute(f"SET statement_timeout = {self.statement_timeout};")
            self.__db.commit()
        return res

    def result(self, sql):
        assert self.__db is not None
        return iterator_utils.cursor_iter(self.__db, sql)

    def table_columns(self, table_name, schema_name=None):
        assert self.__db is not None
        if schema_name is None:
            schema_name = 'public'
        return list(map(lambda x: x[1], postgres.table_structure(self.__db, table_name, schema_name=schema_name)))

    def __latency(self, sql, cache=True):
        res = self.plan_latency(sql, cache=cache)
        cost = res[0][0][0]['Plan']['Actual Total Time']
        return cost

    def latency(self, sql, origin=None, cache=True, throw=False):
        assert self.__db is not None
        if cache and sql in self.__latency_cache:
            return self.__latency_cache[sql]
        timeout_limit = self.__timeout_limit(sql)
        self.__pre_settings()

        if origin is None:
            latency = None
            try:
                latency = self.__latency(sql, cache=cache)
            except Exception as e:
                self.__db.commit()
                if throw:
                     raise e
            if latency is None:
                latency = timeout_limit
            self.__latency_cache[sql] = latency
            self.__auto_save()
            return latency

        cost = self.cost(sql)
        cost_origin = self.cost(origin)

        latency = None
        if cost / cost_origin < 100:
            try:
                latency = self.__latency(sql, cache=cache)
            except Exception as e:
                self.__db.commit()
                if throw:
                    raise e
        if latency is None:
            latency = min(cost / cost_origin * self.latency(origin, cache=cache), timeout_limit)

        self.__latency_cache[latency] = latency
        self.__auto_save()
        return latency

    def __timeout_limit(self, sql):
        assert self.__db is not None
        if sql in self.__latency_cache:
            return self.__latency_cache[sql] * 4 + self.plan_time(sql)
        return self.statement_timeout

    @property
    def schema(self):
        return self.__schema

database = _global()


def postgres_type(type_name):
    if type_name in (
        'bigint', 'int8',
        'bigserial', 'serial8',
        'integer', 'int', 'int4',
        'smallint', 'int2',
        'smallserial', 'serial2',
        'serial', 'serial4',
    ):
        return 1
    if type_name in (
        'double precision', 'float8',
        'numeric', 'decimal',
        'real', 'float4',
    ):
        return 2
    return 0 # non-numeric types


class DataTable:
    def __init__(self, struct, name, schemaname = 'public'):
        self.columns = {}
        self.column_indexes = {}
        self.column_types = {}
        self.name = name
        self.schemaname = schemaname
        self.size = len(struct)
        self.row_count = 0
        for index, (i, name, typ, *_) in enumerate(struct):
            self.columns[index] = name
            self.column_indexes[name] = index
            self.column_types[name] = typ

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls((), None)
        obj.columns = state_dict['columns']
        obj.column_indexes = state_dict['column_indexes']
        obj.name = state_dict['name']
        obj.schemaname = state_dict['schemaname']
        obj.size = state_dict['size']
        return obj

    def state_dict(self):
        return {
            'columns': self.columns,
            'column_indexes': self.column_indexes,
            'name': self.name,
            'schemaname': self.schemaname,
            'size': self.size,
        }

    @property
    def count(self):
        return database.table_size(self.name)

    def __len__(self):
        return self.size


class Schema:
    def __init__(self, db: postgres.Connection, database: _global):
        schema_cache_file = f'sql/{database.name}.schema.pkl'
        if os.path.isfile(schema_cache_file):
            with open(schema_cache_file, 'rb') as f:
                state_dict = pickle.load(f)
            self.load_state_dict(state_dict)
        else:
            assert db is not None, f'Failed to load schema cache file \'{schema_cache_file}\'.'

            tables = postgres.tables(db)

            tables = list(filter(lambda x: x[0] == 'public', tables))

            self.tables = []
            self.name_to_indexes = {}
            self.table_names = []
            self.name_to_table = {}
            self.size = len(tables)

            self.table_structures = {}

            self.total_columns = 0
            self.max_columns = 0

            self.columns = []
            self.column_indexes = {}
            self._column_features = []
            self.column_features_size = 0
            self.column_db_indexes = {}

            for i, (sname, tname) in enumerate(tables):
                table_obj = postgres.table_structure(db, tname, sname)
                self.max_columns = max(self.max_columns, len(table_obj))

                for j, name, typ, *_ in table_obj:
                    c = (sname, tname, name)
                    self.column_indexes[c] = len(self.columns)
                    self.columns.append(c)
                    self.column_db_indexes[c] = []

                table_obj = DataTable(table_obj, tname, sname)
                self.tables.append(table_obj)
                self.table_names.append(tname)
                self.name_to_indexes[tname] = i
                self.name_to_table[tname] = table_obj
                self.total_columns += len(table_obj)

                row_count_query = f'select count(*) from {sname + "." if sname else ""}{tname};'
                row_count = database.first_element(row_count_query)
                table_obj.row_count = row_count

                index_query = f'select indexname, indexdef from pg_indexes ' \
                              f'where schemaname = \'{"public" if sname is None else sname}\' and tablename = \'{tname}\';'
                res = list(database.result(index_query))
                for index_name, index_def in res:
                    match = re.search(r'\(([A-Za-z0-9_]+(?:, *[A-Za-z0-9_]+)*)\)', index_def)
                    if match:
                        columns = match.group(0)
                        columns = re.findall(r'[A-Za-z0-9_]+', columns)
                        for c in columns:
                            c = (sname, tname, c)
                            assert c in self.column_db_indexes
                            self.column_db_indexes[c].append((index_name, index_def))

            cache_file = f'sql/{database.name}.schema_cache.pkl'
            if os.path.isfile(cache_file):
                with open(cache_file, 'rb') as f:
                    self._column_features = pickle.load(f)
            else:
                print('Calculating column features', file=sys.stderr)

                for index, tup in tqdm(enumerate(self.columns), total=len(self.columns)):
                    sname, tname, cname = tup
                    table_obj = self.name_to_table[tname]

                    # properties
                    #   - one-hot encoding of column type (is or is not string type)
                    #   - one-hot encoding of inferred value type (type-like or common or one-hot-like) (n_distinct)
                    #   - whether the column is likely to increase n_distinct (n_distinct)
                    #   - whether contains null values (null_frac)
                    #   - whether the column has index

                    column_type = postgres_type(table_obj.column_types[cname])
                    column_type_onehot = [0 for i in range(3)]
                    column_type_onehot[column_type] = 1

                    pg_stats_query = f'select null_frac, n_distinct from pg_stats ' \
                                     f'where schemaname = \'{"public" if sname is None else sname}\' ' \
                                     f'and tablename = \'{tname}\' ' \
                                     f'and attname = \'{cname}\';'
                    res = list(database.result(pg_stats_query))
                    if not res:
                        # not enough data
                        null_frac = 0.0
                        n_distinct = -1.0
                    else:
                        null_frac, n_distinct = res[0]
                    table_row_count = table_obj.row_count

                    is_increasing = n_distinct < 0
                    if is_increasing:
                        real_n_distinct = -n_distinct
                    else:
                        real_n_distinct = n_distinct / table_row_count

                    if real_n_distinct < 0.001:
                        # type-like column
                        row_type = 0
                    elif real_n_distinct < 0.999:
                        # common type
                        row_type = 1
                    else:
                        row_type = 2

                    n_distinct_onehot = [*(0 for i in range(3)), 1 if is_increasing else 0, 0 if is_increasing else 1]
                    n_distinct_onehot[row_type] = 1

                    column_indexes = self.column_db_indexes[tup]
                    column_index_onehot = [1 if column_indexes else 0, 0 if column_indexes else 1]

                    null_frac_onehot = [1 if null_frac == 0 else 0, 1 if 0 < null_frac < 0.5 else 0, 1 if null_frac >= 0.5 else 0]

                    self._column_features.append([
                        *column_type_onehot, # 3
                        *n_distinct_onehot, # 5
                        *null_frac_onehot, # 3
                        *column_index_onehot, # 2
                    ])
                with filepath.position(os.path.dirname(cache_file)):
                    with open(os.path.basename(cache_file), 'wb') as f:
                        pickle.dump(self._column_features, f)
            self.column_features_size = self.total_columns + 2
            with open(schema_cache_file, 'wb') as f:
                pickle.dump(self.state_dict(), f)

    def load_state_dict(self, state_dict):
        tables = state_dict['tables']
        self.tables = []
        for t in tables:
            self.tables.append(DataTable.from_state_dict(t))
        for name in (
            'name_to_indexes', 'table_names', 'name_to_table', 'size', 'table_structures',
            'total_columns', 'max_columns', 'columns', 'column_indexes', '_column_features',
            'column_features_size',
        ):
            setattr(self, name, state_dict[name])

    def state_dict(self):
        res = {}
        for name in (
            'name_to_indexes', 'table_names', 'name_to_table', 'size', 'table_structures',
            'total_columns', 'max_columns', 'columns', 'column_indexes', '_column_features',
            'column_features_size',
        ):
            res[name] = getattr(self, name)
        res['tables'] = [i.state_dict() for i in self.tables]
        return res

    def column_index(self, table_name, column_name, schema_name='public'):
        return self.column_indexes[(schema_name, table_name, column_name)]

    def column_features(self, table_name, column_name, schema_name='public', *,
                        dtype=torch.float, device=torch.device('cpu')):
        column_index = self.column_index(table_name, column_name, schema_name)
        if dtype is not None:
            return torch.tensor(self._column_features[column_index], dtype=dtype, device=device)
        return list(self._column_features[column_index])

    def table_onehot(self, name, *, dtype=torch.float, device=torch.device('cpu')):
        res = torch.zeros(self.size, dtype=dtype, device=device)
        if name is not None:
            res[self.name_to_indexes[name]] = 1
        return res

    @property
    def table_join_size(self):
        return self.size * (self.size + 1) // 2

    def table_join_onehot(self, left, right, *, dtype=torch.float, device=torch.device('cpu'), detail=False):
        res = torch.zeros(self.table_join_size, dtype=dtype, device=device)
        left_index = self.name_to_indexes[left]
        right_index = self.name_to_indexes[right]
        rev = left_index > right_index
        if rev:
            left_index, right_index = right_index, left_index
        index = (self.size * 2 - left_index + 1) * left_index // 2 + right_index - left_index
        res[index] = 1
        if detail:
            return res, rev
        return res

    def __len__(self):
        return self.size

    def table(self, name):
        return self.name_to_indexes[name], self.name_to_table[name]


def sql_expr_analyze(arg, table_space = None):
    """
    Recursive analysis of select statements.
    """
    if 'ColumnRef' in arg:
        return ColumnRef(arg['ColumnRef'], table_space)
    elif 'FuncCall' in arg:
        return FuncCall(arg['FuncCall'], table_space)
    elif 'GroupingFunc' in arg:
        return FuncCall(arg['GroupingFunc'], table_space, name='grouping')
    elif 'CoalesceExpr' in arg:
        return FuncCall(arg['CoalesceExpr'], table_space, name='coalesce')
    elif 'A_Expr' in arg:
        return MathExpr(arg['A_Expr'], table_space)
    elif 'A_Const' in arg:
        return Const(arg['A_Const'])
    elif 'TypeCast' in arg:
        return TypeCast(arg['TypeCast'], table_space)
    elif 'BoolExpr' in arg:
        return BoolExpr(arg['BoolExpr'], table_space)
    elif 'NullTest' in arg:
        return NullTest(arg['NullTest'], table_space)
    elif 'SubLink' in arg:
        return SubLink(arg['SubLink'], table_space)
    elif 'WindowDef' in arg:
        return WindowDef(arg['WindowDef'], table_space)
    elif 'CaseExpr' in arg:
        return CaseExpr(arg['CaseExpr'], table_space)
    elif 'GroupingSet' in arg:
        return GroupingSet(arg['GroupingSet'], table_space)
    else:
        raise Exception(f'Unknown expression: {arg}')


class Element:
    """
    Parent class of all tree elements.
    """
    def oracle(self):
        return str(self)

    @property
    def concerned_columns(self):
        return set()

    @property
    def concerned_aliases(self):
        return set()

class CaseWhen(Element):
    def __init__(self, args, table_space=None):
        self.expr = sql_expr_analyze(args['expr'], table_space)
        self.result = sql_expr_analyze(args['result'], table_space)

    def to_str(self, oracle=False):
        if oracle:
            return f'when {self.expr.oracle()} then {self.result.oracle()}'
        # TODO: do expr and result need parenthesises?
        return f'when {self.expr} then {self.result}'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

    @property
    def concerned_columns(self):
        return self.expr.concerned_columns | self.result.concerned_columns

    @property
    def concerned_aliases(self):
        return self.expr.concerned_aliases | self.result.concerned_aliases

class CaseExpr(Element):
    def __init__(self, args, table_space=None):
        if 'defresult' in args:
            self.default = sql_expr_analyze(args['defresult'], table_space)
        else:
            self.default = None
        cases = args.get('args', [])
        self.cases = []
        for case in cases:
            self.cases.append(CaseWhen(case['CaseWhen'], table_space))

    @property
    def concerned_aliases(self):
        res = set()
        if self.default:
            res |= self.default.concerned_aliases
        for case in self.cases:
            res |= case.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        if self.default:
            res |= self.default.concerned_columns
        for case in self.cases:
            res |= case.concerned_columns
        return res

    def to_str(self, oracle=False):
        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str

        if self.default is None:
            default = ''
        else:
            default = f' else {self.default}'
        cases = ' '.join(map(_str, self.cases))
        return f'case {cases}{default} end'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class WindowDef(Element):
    def __init__(self, args, table_space=None):
        partition = args.get('partitionClause', [])
        self.partition = []
        for element in partition:
            self.partition.append(sql_expr_analyze(element, table_space))
        order = args.get('orderClause', [])
        self.order = []
        for element in order:
            self.order.append(OrderClause(element, table_space))

    @property
    def concerned_aliases(self):
        res = set()
        for element in self.partition:
            res |= element.concerned_aliases
        for element in self.order:
            res |= element.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        for element in self.partition:
            res |= element.concerned_columns
        for element in self.order:
            res |= element.concerned_columns
        return res

    def to_str(self, oracle=False):
        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str

        if self.partition:
            partition = f'partition by {", ".join(map(_str, self.partition))}'
        else:
            partition = ''
        if self.order:
            order = f'order by {", ".join(map(_str, self.order))}'
        else:
            order = ''
        return f'{partition} {order}'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class NullTest(Element):
    def __init__(self, expr, table_space=None):
        arg = expr['arg']
        self.type = expr['nulltesttype']
        self.element = sql_expr_analyze(arg, table_space)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    def __str__(self):
        if self.type == 0:
            type_str = 'IS NULL'
        elif self.type == 1:
            type_str = 'IS NOT NULL'
        else:
            raise Exception(f'Unknown null test type: {self.type}')
        return f'{self.element} {type_str}'

class ColumnRef(Element):
    def __init__(self, col, table_space=None):
        fields = col['fields']
        if len(fields) == 1:
            self.star = 'A_Star' in fields[0]
            if not self.star:
                self.column_name = fields[0]['String']['str']
                self.alias = None
                if table_space:
                    for alias, table in table_space:
                        if self.column_name in table.column_indexes:
                            self.alias = alias
                            break
            else:
                self.column_name = None
                self.alias = None
        else:
            self.alias = fields[0]['String']['str']
            self.star = 'A_Star' in fields[1]
            if not self.star:
                self.column_name = fields[1]['String']['str']
            else:
                self.column_name = None

    @property
    def concerned_aliases(self):
        return {self.alias} if self.alias else set()

    @property
    def concerned_columns(self):
        return {(self.alias, self.column_name)} if self.alias and self.column_name else set()

    def __str__(self):
        if self.alias:
            if self.star:
                return f'{self.alias}.*'
            return f'{self.alias}.{self.column_name}'
        if self.star:
            return '*'
        return self.column_name

class Const(Element):
    def __init__(self, const):
        self.type = None
        value = const["val"]
        if "String" in value:
            self.type = str
            self.value = value["String"]["str"]
        elif "Integer" in value:
            self.type = int
            self.value = value["Integer"]["ival"]
        elif "Float" in value:
            self.type = float
            self.value = float(value["Float"]["str"])
        elif "Null" in value:
            self.type = None
            self.value = None
        else:
            raise Exception("unknown Value in Expr")

    @property
    def concerned_aliases(self):
        return set()

    @property
    def concerned_columns(self):
        return set()

    def __str__(self):
        if self.type == str:
            return f"'{self.value}'"
        if self.value is None:
            return 'null'
        return str(self.value)

class BoolExpr(Element):
    def __init__(self, expr, table_space=None):
        self.op = expr['boolop']
        args = expr['args']
        self.args = []
        for arg in args:
            self.args.append(sql_expr_analyze(arg, table_space))

    @property
    def concerned_aliases(self):
        res = set()
        for arg in self.args:
            res |= arg.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        for arg in self.args:
            res |= arg.concerned_columns
        return res

    def to_str(self, oracle=False):
        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str

        if self.op == 0:
            return ' AND '.join(map(_str, self.args))
        elif self.op == 1:
            return ' OR '.join(map(_str, self.args))
        elif self.op == 2:
            return 'NOT ' + _str(self.args[0])
        else:
            raise Exception('Unknown bool expression')

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class TypeCast(Element):
    def __init__(self, expr, table_space = None):
        self.arg = sql_expr_analyze(expr['arg'])
        type_name = expr['typeName']['TypeName']
        if len(type_name['names']) == 1:
            self.type_class = None
            self.name = type_name['names'][0]['String']['str']
        else:
            self.type_class = type_name['names'][0]['String']['str']
            self.name = type_name['names'][1]['String']['str']
        self.type_args = []
        if 'typmods' in type_name:
            for dic in type_name['typmods']:
                self.type_args.append(sql_expr_analyze(dic, table_space))

    @property
    def concerned_aliases(self):
        res = self.arg.concerned_aliases
        for arg in self.type_args:
            res |= arg.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = self.arg.concerned_columns
        for arg in self.type_args:
            res |= arg.concerned_columns
        return res

    def __str__(self):
        type_name = f'{self.type_class}.{self.name}' if self.type_class and self.type_class != 'pg_catalog' else self.name
        type_args = f'({", ".join(map(str, self.type_args))})' if self.type_args else ''
        return str(self.arg) + '::' + type_name + type_args

    def oracle(self):
        type_name = f'{self.type_class}.{self.name}' if self.type_class and self.type_class != 'pg_catalog' else self.name
        if type_name == 'date':
            return f'to_date({str(self.arg)}, \'YYYY-MM-DD\')'
        return str(self.arg)

class MathExpr(Element):
    def __init__(self, expr, table_space=None):
        self.kind = expr['kind']
        self.name = expr['name'][0]['String']['str']
        if 'lexpr' in expr:
            lexpr = expr['lexpr']
            if isinstance(lexpr, list):
                self.lexpr = []
                for arg in lexpr:
                    self.lexpr.append(sql_expr_analyze(arg, table_space))
            else:
                self.lexpr = sql_expr_analyze(lexpr, table_space)
        else:
            self.lexpr = None
        if 'rexpr' in expr:
            rexpr = expr['rexpr']
            if isinstance(rexpr, list):
                self.rexpr = []
                for arg in rexpr:
                    self.rexpr.append(sql_expr_analyze(arg, table_space))
            else:
                self.rexpr = sql_expr_analyze(rexpr, table_space)
        else:
            self.rexpr = None

    @property
    def concerned_aliases(self):
        res = set()
        if self.lexpr:
            if isinstance(self.lexpr, list):
                for expr in self.lexpr:
                    res |= expr.concerned_aliases
            else:
                res |= self.lexpr.concerned_aliases
        if self.rexpr:
            if isinstance(self.rexpr, list):
                for expr in self.rexpr:
                    res |= expr.concerned_aliases
            else:
                res |= self.rexpr.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        if self.lexpr:
            if isinstance(self.lexpr, list):
                for expr in self.lexpr:
                    res |= expr.concerned_columns
            else:
                res |= self.lexpr.concerned_columns
        if self.rexpr:
            if isinstance(self.rexpr, list):
                for expr in self.rexpr:
                    res |= expr.concerned_columns
            else:
                res |= self.rexpr.concerned_columns
        return res

    def to_str(self, oracle=False):
        if oracle:
            if isinstance(self.lexpr, list):
                lexpr = f'({", ".join(map(lambda x: x.oracle(), self.lexpr))})'
            else:
                lexpr = self.lexpr.oracle()
            if self.kind in (10, 11):
                rexpr = f'{self.rexpr[0].oracle()} AND {self.rexpr[1].oracle()}'
            else:
                if isinstance(self.rexpr, list):
                    rexpr = f'({", ".join(map(lambda x: x.oracle(), self.rexpr))})'
                else:
                    rexpr = self.rexpr.oracle()
        else:
            if isinstance(self.lexpr, list):
                lexpr = f'({", ".join(map(str, self.lexpr))})'
            else:
                lexpr = str(self.lexpr)
            if self.kind in (10, 11):
                rexpr = f'{self.rexpr[0]} AND {self.rexpr[1]}'
            else:
                if isinstance(self.rexpr, list):
                    rexpr = f'({", ".join(map(str, self.rexpr))})'
                else:
                    rexpr = str(self.rexpr)

        if self.kind == 7:
            # like
            if self.name == '!~~':
                return f'{lexpr} NOT LIKE {rexpr}'
            return f'{lexpr} LIKE {rexpr}'
        if self.kind == 8:
            # ilike
            if self.name == '!~~*':
                return f'{lexpr} NOT ILIKE {rexpr}'
            return f'{lexpr} ILIKE {rexpr}'
        if self.kind == 6:
            if self.name == '<>':
                return f'{lexpr} NOT IN {rexpr}'
            return f'{lexpr} IN {rexpr}'
        if self.kind == 10:
            return f'{lexpr} BETWEEN {rexpr}'
        if self.kind == 11:
            return f'{lexpr} NOT BETWEEN {rexpr}'
        assert self.kind == 0, f'Unknown operator {self.name}'
        return f'{lexpr} {self.name} {rexpr}'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class SubLink(Element):
    def __init__(self, sublink, table_space=None):
        self.type = sublink['subLinkType']
        if 'testexpr' in sublink:
            self.test_expr = sql_expr_analyze(sublink['testexpr'])
            self.op = sublink['operName'][0]['String']['str']
        else:
            self.test_expr = None
            self.op = None
        self.subselect_raw = sublink['subselect']
        self.subselect = Sql(self.subselect_raw, table_space=table_space)

    def is_column(self):
        return False

    @property
    def concerned_aliases(self):
        res = self.test_expr.concerned_aliases if self.test_expr else set()
        res |= self.subselect.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = self.test_expr.concerned_columns if self.test_expr else set()
        res |= self.subselect.concerned_columns
        return res

    def to_str(self, oracle=False):
        if self.type == 0:
            sublink = 'exists'
        elif self.type == 1:
            sublink = 'all'
        elif self.type == 2:
            sublink = 'any'
        elif self.type == 4:
            sublink = ''
        else:
            raise Exception(f'Unknown sublink type: {self.type}')
        if self.test_expr is not None:
            if oracle:
                test_expr = self.test_expr.oracle()
            else:
                test_expr = str(self.test_expr)
            return f'{test_expr} {self.op} {sublink}({self.subselect})'
        return f'{sublink}({self.subselect})'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class FuncCall(Element):
    def __init__(self, func_call, table_space=None, name=None):
        if name is None:
            if len(func_call['funcname']) == 1:
                self.name = func_call['funcname'][0]['String']['str']
                self.class_name = None
            else:
                self.name = func_call['funcname'][1]['String']['str']
                self.class_name = func_call['funcname'][0]['String']['str']
        else:
            self.name = name
            self.class_name = None
        self.star = 'agg_star' in func_call
        self.args = []
        if not self.star:
            args = func_call.get('args', [])
            for arg in args:
                self.args.append(sql_expr_analyze(arg, table_space))
        if len(self.args) == 1 and isinstance(self.args[0], ColumnRef):
            self.column_ref = self.args[0]
        else:
            self.column_ref = None
        if 'over' in func_call:
            self.over = sql_expr_analyze(func_call['over'], table_space)
        else:
            self.over = None

    @property
    def concerned_aliases(self):
        res = self.over.concerned_aliases if self.over else set()
        for arg in self.args:
            res |= arg.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = self.over.concerned_columns if self.over else set()
        for arg in self.args:
            res |= arg.concerned_columns
        return res

    @property
    def alias(self):
        if self.column_ref:
            return self.column_ref.alias
        return None

    @property
    def column_name(self):
        if self.column_ref:
            return self.column_ref.column_name
        return None

    def to_str(self, oracle=False):
        if self.class_name is not None:
            name = f'{self.class_name}.{self.name}'
        else:
            name = self.name
        if self.over is not None:
            over = f' over ({self.over})'
        else:
            over = ''
        if self.star:
            return f'{name}(*){over}'
        if oracle:
            return f'{name}({", ".join(map(lambda x: x.oracle(), self.args))}){over}'
        return f'{name}({", ".join(map(str, self.args))}){over}'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class TargetTable(Element):
    def __init__(self, target, table_space=None):
        self.target = target
        arg = self.target['val']
        self.name = self.target.get('name', None)
        self.element = sql_expr_analyze(arg, table_space)
        self.column_ref = isinstance(self.element, ColumnRef)
        self.func_call = isinstance(self.element, FuncCall)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    @property
    def res_name(self):
        if self.name is not None:
            return self.name
        if self.func_call:
            if self.element.class_name is None:
                return self.element.name
            return f'{self.element.class_name}.{self.element.name}'
        if self.column_ref:
            return self.element.column_name
        return '?'

    @property
    def star(self):
        if self.func_call:
            return self.element.star
        if self.column_ref:
            return self.element.star
        return False

    @property
    def value(self):
        if self.func_call:
            self.element : FuncCall
            if self.element.column_ref is not None:
                return str(self.element.column_ref)
            return None
        if self.column_ref:
            return str(self.element)
        return None

    @property
    def alias(self):
        if self.func_call:
            if self.element.column_ref is not None:
                return self.element.column_ref.alias
        if self.column_ref:
            return self.element.alias
        return None

    @property
    def column_name(self):
        if self.func_call:
            if self.element.column_ref is not None:
                return self.element.column_ref.column_name
        if self.column_ref:
            return self.element.column_name
        return None

    def to_str(self, oracle=False):
        if self.name is not None:
            if not re.match('[A-Za-z_][A-Za-z0-9_]*', self.name):
                name = f'"{self.name}"'
            else:
                name = self.name
            as_clause = f' AS {name}'
        else:
            as_clause = ''
        if oracle:
            return self.element.oracle() + as_clause
        return str(self.element) + as_clause

    def oracle(self):
        return self.to_str(True)

    def __str__(self):
        return self.to_str(False)


class FromTable(Element):
    def __init__(self, from_table):
        self.from_table = from_table

    @property
    def concerned_aliases(self):
        return {self.alias}

    @property
    def concerned_columns(self):
        return set()

    @property
    def fullname(self):
        return self.from_table["relname"]

    @property
    def alias(self):
        if "alias" in self.from_table:
            return self.from_table["alias"]["Alias"]["aliasname"]
        return self.fullname

    def __str__(self):
        return self.fullname + " AS " + self.alias

    def oracle(self):
        return self.oracle_str()

    def oracle_str(self):
        return self.fullname + " " + self.alias

class Subquery(Element):
    def __init__(self, subquery, table_space=None):
        sql = subquery['subquery']
        self.alias = subquery['alias']['Alias']['aliasname']
        column_names = subquery.get('colnames', None)
        if column_names:
            targets = sql['SelectStmt']['targetList']
            for target, column_name in zip(targets, column_names):
                target['ResTarget']['name'] = column_name
        self.subquery = Sql(sql, table_space=table_space)

    @property
    def concerned_aliases(self):
        return self.subquery.concerned_aliases | {self.alias}

    @property
    def concerned_columns(self):
        return self.subquery.concerned_columns

    def __str__(self):
        return f'({str(self.subquery)}) {self.alias}'

class GroupingSet(Element):
    def __init__(self, args, table_space=None):
        self.kind = args['kind']
        content = args.get('content', [])
        self.content = []
        for element in content:
            self.content.append(sql_expr_analyze(element, table_space))

    @property
    def concerned_aliases(self):
        res = set()
        for arg in self.content:
            res |= arg.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        for arg in self.content:
            res |= arg.concerned_columns
        return res

    def __str__(self):
        if self.kind == 2:
            name = 'rollup'
        elif self.kind == 3:
            name = 'cube'
        elif self.kind == 4:
            name = 'grouping sets'
        else:
            name = ''
        args = ', '.join(map(str, self.content))
        return f'{name}({args})'

class GroupClause(Element):
    def __init__(self, clause, table_space=None):
        self.element = sql_expr_analyze(clause, table_space)
        self.column_ref = isinstance(self.element, ColumnRef)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    @property
    def alias(self):
        if self.column_ref:
            return self.element.alias
        return None

    @property
    def column_name(self):
        if self.column_ref:
            return self.element.column_name
        return None

    def __str__(self):
        return str(self.element)

    def oracle(self):
        return self.element.oracle()

class OrderClause(Element):
    def __init__(self, order, table_space=None):
        o = order['SortBy']
        self.direction = o['sortby_dir']
        node = o['node']
        self.element = sql_expr_analyze(node, table_space)
        self.column_ref = isinstance(self.element, ColumnRef)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    @property
    def alias(self):
        if self.column_ref:
            return self.element.alias
        return None

    @property
    def column_name(self):
        if self.column_ref:
            return self.element.column_name
        return None

    def __str__(self):
        if self.direction == 1:
            direction = ' asc'
        elif self.direction == 2:
            direction = ' desc'
        else:
            direction = ''
        return f'{str(self.element)}{direction}'

    def oracle(self):
        if self.direction == 1:
            direction = ' asc'
        elif self.direction == 2:
            direction = ' desc'
        else:
            direction = ''
        return f'{self.element.oracle()}{direction}'

class LimitClause(Element):
    def __init__(self, clause, table_space=None):
        self.element = sql_expr_analyze(clause, table_space)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    def __str__(self):
        return f'LIMIT {self.element}'

    def oracle(self):
        return f'rownum <= {self.element}'

class HavingClause(Element):
    def __init__(self, clause, table_space=None):
        self.element = sql_expr_analyze(clause, table_space)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    def __str__(self):
        return f'HAVING {self.element}'

class Expr(Element):
    def __init__(self, expr, list_kind = 0, table_space=None):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.val = 0
        if isinstance(expr, list):
            self.element = [sql_expr_analyze(e, table_space) for e in expr]
        else:
            self.element = sql_expr_analyze(self.expr, table_space)
        if isinstance(self.element, ColumnRef):
            self.column_ref = self.element
        else:
            self.column_ref = None

    @property
    def concerned_aliases(self):
        if isinstance(self.element, list):
            res = set()
            for element in self.element:
                res |= element.concerned_aliases
            return res
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        if isinstance(self.element, list):
            res = set()
            for element in self.element:
                res |= element.concerned_columns
            return res
        return self.element.concerned_columns

    @property
    def alias(self):
        if self.column_ref:
            return self.column_ref.alias
        return None

    @property
    def column_name(self):
        if self.column_ref:
            return self.column_ref.column_name
        return None

    def is_column(self):
        return self.column_ref is not None

    def is_numeric(self, value_expr=None):
        if value_expr is None:
            value_expr = self.expr
            if isinstance(value_expr, list):
                value_expr = value_expr[0]
        if "A_Const" in value_expr:
            value = value_expr["A_Const"]["val"]
            if "Integer" in value or "Float" in value:
                return True
        return False

    def get_value(self, value_expr=None, _int=False, oracle=False):
        if value_expr is None:
            value_expr = self.element#self.expr
        if isinstance(value_expr, list):
            return [self.get_value(i, _int=_int, oracle=oracle) for i in value_expr]
        if isinstance(value_expr, Const):
            if _int:
                return value_expr.value
            return str(value_expr)
        elif isinstance(value_expr, TypeCast):
            if oracle:
                return value_expr.oracle()
            return str(value_expr)
        else:
            return str(value_expr)

    def __str(self, oracle=False):
        if self.is_column():
            return str(self.column_ref)
        elif isinstance(self.expr, dict):
            return self.get_value(self.element, oracle=oracle)
        elif isinstance(self.expr, list):
            if self.list_kind == 6:
                return "(" +",\n".join([self.get_value(x, oracle=oracle) for x in self.element]) + ")"
            elif self.list_kind in (10, 11):
                return " AND ".join([self.get_value(x, oracle=oracle) for x in self.element])
            else:
                raise Exception("list kind error")
        else:
            raise Exception("No Known type of Expr")

    def __str__(self):
        return self.__str()

    def oracle(self):
        return self.__str(oracle=True)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return str(self) == str(other)
        return False


class Comparison(Element):
    """
    Originate from RTOS.
    """
    def __init__(self, comparison, table_space=None):
        self.comparison = comparison
        self.column_list = []
        self.aliasname_list = []
        self.op = None
        self.lexpr = None
        self.rexpr = None
        if "A_Expr" in self.comparison:
            self.lexpr = Expr(comparison["A_Expr"]["lexpr"], table_space=table_space)
            self.kind = comparison["A_Expr"]["kind"]
            self.op = self.comparison["A_Expr"]["name"][0]["String"]["str"]

            if not "A_Expr" in comparison["A_Expr"]["rexpr"]:
                self.rexpr = Expr(comparison["A_Expr"]["rexpr"],self.kind, table_space=table_space)
            else:
                self.rexpr = Comparison(comparison["A_Expr"]["rexpr"], table_space=table_space)

            l_concerned = self.lexpr.concerned_columns
            r_concerned = self.rexpr.concerned_columns

            for t, c in l_concerned:
                self.aliasname_list.append(t)
                self.column_list.append(c)
            for t, c in r_concerned - l_concerned:
                self.aliasname_list.append(t)
                self.column_list.append(c)

            self.comp_kind = 0

        elif "NullTest" in self.comparison:
            self.lexpr = Expr(comparison["NullTest"]["arg"], table_space=table_space)
            self.kind = comparison["NullTest"]["nulltesttype"]

            l_concerned = self.lexpr.concerned_columns
            for t, c in l_concerned:
                self.aliasname_list.append(t)
                self.column_list.append(c)

            self.comp_kind = 1

        elif "BoolExpr" in self.comparison:
            self.kind = comparison["BoolExpr"]["boolop"]
            self.comp_list = [Comparison(x, table_space=table_space)
                              for x in comparison["BoolExpr"]["args"]]

            for comp in self.comp_list:
                if comp.comp_kind == 2:
                    self.aliasname_list.extend(comp.aliasname_list)
                    self.column_list.extend(comp.column_list)
                else:
                    l_concerned = comp.lexpr.concerned_columns
                    r_concerned = comp.rexpr.concerned_columns if comp.rexpr else set()
                    for t, c in l_concerned:
                        self.aliasname_list.append(t)
                        self.column_list.append(c)
                    for t, c in r_concerned - l_concerned:
                        self.aliasname_list.append(t)
                        self.column_list.append(c)

            self.comp_kind = 2

        elif "SubLink" in self.comparison:
            self.comp_kind = 3
            self.lexpr = SubLink(comparison['SubLink'], table_space=table_space)

            l_concerned = self.lexpr.concerned_columns
            for t, c in l_concerned:
                self.aliasname_list.append(t)
                self.column_list.append(c)
        else:
            raise Exception(f'Unknown comparison type: {comparison}')

        self.aliasname_set = set(self.aliasname_list)

    @property
    def concerned_aliases(self):
        res = set()
        if self.comp_kind == 2:
            for c in self.comp_list:
                res |= c.concerned_aliases
        else:
            if self.lexpr is not None:
                res |= self.lexpr.concerned_aliases
            if self.rexpr is not None:
                res |= self.rexpr.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        if self.comp_kind == 2:
            for c in self.comp_list:
                res |= c.concerned_columns
        else:
            if self.lexpr is not None:
                res |= self.lexpr.concerned_columns
            if self.rexpr is not None:
                res |= self.rexpr.concerned_columns
        return res

    @property
    def left_column_name(self):
        if isinstance(self.lexpr, Expr):
            return self.lexpr.column_name
        concerned = self.lexpr.concerned_columns
        assert len(concerned) == 1
        return list(self.lexpr.concerned_columns)[0][1]

    @property
    def right_column_name(self):
        if isinstance(self.rexpr, Expr):
            return self.rexpr.column_name
        concerned = self.rexpr.concerned_columns
        assert len(concerned) == 1
        return list(self.rexpr.concerned_columns)[0][1]

    def is_numeric(self):
        return False

    def is_column(self):
        return False

    def to_str(self, left_rename=None, right_rename=None, oracle=False):
        if left_rename is not None:
            lexpr = left_rename
        else:
            if oracle and self.lexpr is not None:
                lexpr = self.lexpr.oracle()
            else:
                lexpr = str(self.lexpr)

        if self.comp_kind == 0:
            if right_rename is not None and self.rexpr.is_column():
                rexpr = right_rename
            else:
                if oracle and self.rexpr is not None:
                    rexpr = self.rexpr.oracle()
                else:
                    rexpr = str(self.rexpr)

            Op = ""
            if self.kind == 0:
                Op = self.comparison["A_Expr"]["name"][0]["String"]["str"]
            elif self.kind == 7:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"]=="!~~":
                    Op = "not like"
                else:
                    Op = "like"
            elif self.kind == 8:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"] == "!~~*":
                    Op = "not ilike"
                else:
                    Op = "ilike"
            elif self.kind == 6:
                Op = "IN"
            elif self.kind == 10:
                Op = "BETWEEN"
            elif self.kind == 11:
                Op = "NOT BETWEEN"
            else:
                import json
                print(json.dumps(self.comparison, sort_keys=True, indent=4))
                raise Exception("Operation ERROR")
            return lexpr + " " + Op + " " + rexpr
        elif self.comp_kind == 1:
            if self.kind == 1:
                return lexpr+" IS NOT NULL"
            else:
                return lexpr+" IS NULL"
        elif self.comp_kind == 3:
            # sublink
            return self.lexpr.to_str(oracle=oracle)
        else:
            if self.kind == 2:
                # not
                return f'NOT {self.comp_list[0].to_str(oracle=oracle)}'
            res = ""
            for comp in self.comp_list:
                # TODO: to check later
                comp_str = comp.to_str(left_rename, right_rename, oracle=oracle)

                if res == "":
                    res += "( "+comp_str
                else:
                    if self.kind == 1:
                        res += " OR "
                    else:
                        res += " AND "
                    res += comp_str
            res += ")"
            return res

    def __str__(self):
        return self.to_str()

    def oracle(self):
        return self.to_str(oracle=True)


import os

import math

def sql_repr(value):
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)

def like_to_re_pattern(like):
    res = ['^']
    escape = False
    for c in like:
        if escape:
            if not c in ('%', '_'):
                if c == '\\':
                    res.append('\\')
                else:
                    res.append('\\\\')
            res.append(c)
            escape = False
        elif c == '\\':
            escape = True
        elif c == '%':
            res.append('[\\S\\s]*')
        elif c == '_':
            res.append('[\\S\\s]')
        elif c in '^${}[]().*|?+':
            res.append('\\' + c)
        else:
            res.append(c)
    res.append('$')
    return ''.join(res)

def like_match(pattern, value):
    pattern = like_to_re_pattern(pattern)
    return re.search(pattern, value) is not None

class Sql:
    _re_like = re.compile(r'^%([^%]+)%$')

    def __init__(self, sql, feature_length=2, filename=None, device=torch.device('cpu'), table_space=None):
        self.filename = filename

        self.feature_length = feature_length
        self.sql = sql

        self.is_subquery = isinstance(sql, dict)
        if self.is_subquery:
            self.pre_actions = []
            self.post_actions = []
            parse_result = sql['SelectStmt']
        else:
            parse_result_all = parse_dict(self.sql)

            select = None
            for i, p in enumerate(parse_result_all):
                if select is None and 'SelectStmt' in p:
                    select = i
            assert select is not None, 'No select statement'

            parse_result = parse_dict(self.sql)[select]['SelectStmt']

            sql_arr = list(map(lambda x: x + ';', filter(lambda x: x.strip(), sql.split(';'))))
            self.pre_actions = sql_arr[:select]
            self.post_actions = sql_arr[select + 1:]

        self.from_tables = []

        subqueries = []
        for x in parse_result['fromClause']:
            if 'RangeVar' in x:
                self.from_tables.append(FromTable(x['RangeVar']))
            else:
                subqueries.append(x['RangeSubselect'])

        self.alias_to_table = {table.alias: table for table in self.from_tables}
        self.aliases = set(self.alias_to_table.keys())

        for subquery in subqueries:
            self.aliases.add(subquery['alias']['Alias']['aliasname'])

        data_tables = [(x.alias, database.schema.table(x.fullname)[1]) for x in self.from_tables]
        self.related_subquery = False
        if table_space:
            for alias, table in table_space:
                if not alias in self.aliases:
                    data_tables.append((alias, table))
                    self.related_subquery = True

        for subquery in subqueries:
            subquery = Subquery(subquery, data_tables)
            self.from_tables.append(subquery)
            self.alias_to_table[subquery.alias] = subquery

        self.table_space = data_tables

        self.target_tables = [TargetTable(x["ResTarget"], data_tables) for x in parse_result["targetList"]]

        self.comparisons = [Comparison(x, table_space=data_tables) for x in parse_result["whereClause"]["BoolExpr"]["args"]]\
            if 'whereClause' in parse_result else []

        self.distinct = 'distinctClause' in parse_result
        self.group_clauses = []

        if 'groupClause' in parse_result:
            for clause in parse_result['groupClause']:
                self.group_clauses.append(GroupClause(clause, data_tables))

        self.having_clause = HavingClause(parse_result['havingClause']) if 'havingClause' in parse_result else None

        self.order_clauses = []

        if 'sortClause' in parse_result:
            for clause in parse_result['sortClause']:
                self.order_clauses.append(OrderClause(clause, data_tables))

        self.limit_clause = LimitClause(parse_result['limitCount'], data_tables) if 'limitCount' in parse_result else None

        self.parse_join()

        if not self.is_subquery:
            self._baseline = Baseline(self)
            self.baseline = self._baseline

            self.to(device)

            self.__hetero_graph_dgl = self.to_hetero_graph_dgl(clone=True)

        self.__tail_clauses = {}
        self.tail_clauses()

    @property
    def concerned_aliases(self):
        from_aliases = set()
        for ft in self.from_tables:
            from_aliases |= ft.concerned_aliases

        target_aliases = set()
        for tt in self.target_tables:
            target_aliases |= tt.concerned_aliases

        where_aliases = set()
        for cmp in self.comparisons:
            where_aliases |= cmp.concerned_aliases

        others = [*self.group_clauses, *self.order_clauses]
        if self.having_clause:
            others.append(self.having_clause)
        if self.limit_clause:
            others.append(self.limit_clause)
        other_aliases = set()
        for other in others:
            other_aliases |= other.concerned_aliases

        return (target_aliases | where_aliases | other_aliases) - from_aliases

    @property
    def concerned_columns(self):
        from_aliases = set()
        for ft in self.from_tables:
            from_aliases |= ft.concerned_aliases

        target_columns = set()
        for tt in self.target_tables:
            target_columns |= tt.concerned_columns

        where_columns = set()
        for cmp in self.comparisons:
            where_columns |= cmp.concerned_columns

        others = [*self.group_clauses, *self.order_clauses]
        if self.having_clause:
            others.append(self.having_clause)
        if self.limit_clause:
            others.append(self.limit_clause)
        other_columns = set()
        for other in others:
            other_columns |= other.concerned_columns

        all_columns = target_columns | where_columns | other_columns
        res = set()
        for col in all_columns:
            t, c = col
            if not t in from_aliases:
                res.add(col)

        return res

    def tail_clauses(self, oracle=False):
        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str
        args = (oracle, )
        if args in self.__tail_clauses:
            return self.__tail_clauses[args]
        if self.group_clauses:
            group = f' group by {", ".join(map(_str, self.group_clauses))}'
        else:
            group = ''
        if self.having_clause is not None:
            having = f' having {_str(self.having_clause)}'
        else:
            having = ''
        if self.order_clauses:
            order = f' order by {", ".join(map(_str, self.order_clauses))}'
        else:
            order = ''
        if self.limit_clause is not None and not oracle:
            limit = f' {_str(self.limit_clause)}'
        else:
            limit = ''
        res = f'{group}{having}{order}{limit}'
        self.__tail_clauses[args] = res
        return res

    @property
    def baseline_order(self):
        return self.baseline.result_order

    def __str__(self):
        if isinstance(self.sql, str):
            return self.sql
        return self.__str()

    def oracle(self):
        edges = []
        for left_alias, right_alias, cmp in self.edge_list:
            edges.append(cmp)

        complicated = self.complicated_cmps

        targets = self.target_tables

        from_tables = self.from_tables

        filters = []
        for alias, table_filters in self.filters.items():
            filters.extend(table_filters)

        predicates = edges + filters + complicated

        if predicates:
            where = (
                '\nWHERE ',
                '\nAND '.join(x.oracle() for x in predicates),
            )
        else:
            where = ()

        res = ''.join((
            'SELECT ',
            #'DISTINCT ' if self.distinct else '',
            ',\n'.join(x.oracle() for x in targets),
            '\nFROM ',
            ',\n'.join(x.oracle() for x in from_tables),
            *where,
            '\n',
            self.tail_clauses(oracle=True),
        ))

        if self.limit_clause is not None:
            res = f'SELECT * from ({res}) where {self.limit_clause.oracle()}'

        return res

    def __str(self):
        edges = list(map(lambda x: x[2], self.edge_list))
        filters = []
        for filter_set in self.filters.values():
            filters.extend(filter_set)
        complicated = self.complicated_cmps
        targets = self.target_tables
        from_tables = self.from_tables

        predicates = edges + filters + complicated
        if predicates:
            where = (
                '\nWHERE ',
                '\nAND '.join(str(x) for x in predicates),
            )
        else:
            where = ()

        return ''.join((
            'SELECT ',
            ',\n'.join(str(x) for x in targets),
            '\nFROM ',
            ',\n'.join(str(x) for x in from_tables),
            *where,
            self.tail_clauses(),
            '' if self.is_subquery else ';',
        ))

    def to(self, device):
        self.__device = device
        self.join_matrix = self.join_matrix.to(device)
        self.selectivities = self.selectivities.to(device)
        for attrs in (self.table_feature_filter, self.table_feature_filter_mask, self.table_feature_edge, self.table_feature_global, self.table_feature_onehot, self.table_feature_others):
            for k, v in attrs.items():
                attrs[k] = v.to(device)
        return self

    @property
    def device(self):
        return self.__device

    def parse_join(self):
        self.join_edges = {}
        self.join_matrix = []
        self.filters = {}
        self.join_candidates = set() # used for action selection
        self.edges = set()
        self.edge_list = []

        self.filter_list = []

        self.table_feature_filter = {}
        self.table_feature_filter_mask = {}
        self.table_feature_edge = {}
        self.table_feature_global = {}
        self.table_feature_onehot = {}
        self.table_feature_others = {}

        self.hidden_edge_sets = {}
        self.hidden_edge_union_find_set = {}
        self.hidden_edges = {}

        self.complicated_cmps = []

        for alias in self.aliases:
            self.join_edges[alias] = []
            if isinstance(self.alias_to_table[alias], Subquery):
                # TODO: dummy features for subquery table
                tail_features = [0]

                self.table_feature_filter[alias] = [0.0 for i in range(database.schema.max_columns)]
                self.table_feature_filter_mask[alias] = [0 for i in range(database.schema.max_columns)]

                self.table_feature_edge[alias] = [0.0 for i in range(database.schema.max_columns)]
                self.table_feature_global[alias] = tail_features
                self.table_feature_onehot[alias] = database.schema.table_onehot(None)

                self.table_feature_others[alias] = [[0.0 for i in range(13)] for j in range(database.schema.max_columns)]
            else:
                table_index, table_name, table = self.get_table(alias)
                tail_features = [0]

                self.table_feature_filter[alias] = [0.0 for i in range(database.schema.max_columns)]
                self.table_feature_filter_mask[alias] = [0 for i in range(database.schema.max_columns)]

                self.table_feature_edge[alias] = [0.0 for i in range(database.schema.max_columns)]
                self.table_feature_global[alias] = tail_features
                self.table_feature_onehot[alias] = database.schema.table_onehot(table.name)

                table_feature_others = [[0.0 for i in range(13)] for i in range(database.schema.max_columns)]
                _, _, table = self.get_table(alias)
                for index, column_name in table.columns.items():
                    features = database.schema.column_features(table.name, column_name, dtype=None)
                    table_feature_others[index] = [*features]
                self.table_feature_others[alias] = table_feature_others

        for _ in range(len(database.schema)):
            self.join_matrix.append([0 for i in range(database.schema.size)])

        self.selectivities = [0 for i in range(database.schema.total_columns)]

        for cmp in self.comparisons:
            concerned = cmp.concerned_aliases
            if len(cmp.aliasname_set) == 2 and len(cmp.aliasname_list) == 2:
                # between two columns
                left_alias, right_alias = cmp.aliasname_list

                if left_alias in self.aliases:
                    # TODO: At present, outer aliases in related subquery are ignored
                    left_edge = self.join_edges.setdefault(left_alias, [])
                    left_edge.append((right_alias, cmp))
                    left_is_table = isinstance(self.alias_to_table[left_alias], FromTable)
                else:
                    left_is_table = False
                if right_alias in self.aliases:
                    right_edge = self.join_edges.setdefault(right_alias, [])
                    right_edge.append((left_alias, cmp))
                    right_is_table = isinstance(self.alias_to_table[right_alias], FromTable)
                else:
                    right_is_table = False

                self.edges.add(cmp)
                self.edge_list.append((left_alias, right_alias, cmp))

                if left_is_table:
                    left_table_index, left_table_name, left_table = self.get_table(left_alias)
                if right_is_table:
                    right_table_index, right_table_name, right_table = self.get_table(right_alias)

                if left_is_table and right_is_table:
                    self.join_matrix[left_table_index][right_table_index] = 1
                    self.join_matrix[right_table_index][left_table_index] = 1

                left_column, right_column = cmp.column_list

                if left_is_table:
                    self.table_feature_edge[left_alias][left_table.column_indexes[left_column]] = 1

                if right_is_table:
                    self.table_feature_edge[right_alias][right_table.column_indexes[right_column]] = 1

                self.join_candidates.add((left_alias, right_alias))
                self.join_candidates.add((right_alias, left_alias))

                left_expr_name = str(cmp.lexpr)
                right_expr_name = str(cmp.rexpr)

                left_expr_parent = self._hidden_edge_parent(left_expr_name)
                right_expr_parent = self._hidden_edge_parent(right_expr_name)

                if left_expr_parent == right_expr_parent:
                    # impossible
                    edges = self.hidden_edge_sets.setdefault(left_expr_parent, set())
                    edges.add(cmp.lexpr)
                    edges.add(cmp.rexpr)
                else:
                    edges_left = self.hidden_edge_sets.get(left_expr_parent, None)
                    if edges_left is None:
                        edges_right = self.hidden_edge_sets.setdefault(right_expr_parent, set())
                        edges_right.add(cmp.lexpr)
                        edges_right.add(cmp.rexpr)
                        self.hidden_edge_union_find_set[left_expr_parent] = right_expr_parent
                    else:
                        edges_left.add(cmp.lexpr)
                        edges_left.add(cmp.rexpr)
                        edges_right = self.hidden_edge_sets.get(right_expr_parent, None)
                        if edges_right is not None:
                            edges_left.update(edges_right)
                            self.hidden_edge_sets.pop(right_expr_parent)
                            self.hidden_edge_union_find_set[right_expr_parent] = left_expr_parent
                        self.hidden_edge_union_find_set[right_expr_parent] = left_expr_parent

            elif len(cmp.aliasname_set) == 1 and cmp.comp_kind != 3:
                # between a column and a value, not sublink
                alias = cmp.aliasname_list[0]
                filter = self.filters.setdefault(alias, [])
                filter.append(cmp)
                is_table = isinstance(self.alias_to_table[alias], FromTable)

                if is_table:
                    table_index, table_name, table = self.get_table(alias)
                    selectivity, row_count, total_row_count = database.selectivity(table_name, str(cmp), explain=True, detail=True)

                    for column_name in cmp.column_list:
                        column_index = table.column_indexes[column_name]

                        schema_column_index = database.schema.column_index(table.name, column_name)
                        _log_selectivity = -math.log(selectivity + 1e-9)
                        self.selectivities[schema_column_index] = max(self.selectivities[schema_column_index], _log_selectivity)

                        self.filter_list.append((selectivity, table.name, column_name, alias, cmp))

                        mask = self.table_feature_filter_mask[alias][column_index]
                        if mask:
                            self.table_feature_filter[alias][column_index] = \
                                min(
                                    self.table_feature_filter[alias][column_index],
                                    selectivity,
                                )
                        else:
                            self.table_feature_filter[alias][column_index] = selectivity
                        self.table_feature_filter_mask[alias][column_index] = 1

            elif len(concerned) == 1:
                # complicated single-table predicates
                alias = list(concerned)[0]
                filter = self.filters.setdefault(alias, [])
                filter.append(cmp)
                is_table = isinstance(self.alias_to_table[alias], FromTable)

                if is_table:
                    # TODO: to handle complicated single-table predicates
                    pass
            else:
                if len(cmp.aliasname_set) == 2:
                    left_alias, right_alias = cmp.aliasname_set
                    left_edge = self.join_edges.setdefault(left_alias, [])
                    right_edge = self.join_edges.setdefault(right_alias, [])
                    left_edge.append((right_alias, cmp))
                    right_edge.append((left_alias, cmp))
                    self.edges.add(cmp)
                    self.edge_list.append((left_alias, right_alias, cmp))
                elif len(cmp.aliasname_set) == 1:
                    alias, *_ = cmp.aliasname_set
                    filter = self.filters.setdefault(alias, [])
                    filter.append(cmp)
                else:
                    self.complicated_cmps.append(cmp)

        for alias in self.aliases:
            self.table_feature_filter[alias] = torch.tensor(self.table_feature_filter[alias]).unsqueeze(0).detach()
            self.table_feature_filter_mask[alias] = torch.tensor(self.table_feature_filter_mask[alias]).unsqueeze(0).detach()
            self.table_feature_edge[alias] = torch.tensor(self.table_feature_edge[alias]).unsqueeze(0).detach()
            self.table_feature_global[alias] = torch.tensor(self.table_feature_global[alias]).unsqueeze(0).detach()
            self.table_feature_others[alias] = torch.tensor(self.table_feature_others[alias]).unsqueeze(0).detach()

        self.join_matrix = torch.tensor(np.asarray(self.join_matrix).reshape(1, -1),
                                        dtype=torch.float32)
        self.selectivities = torch.tensor(self.selectivities).unsqueeze(0).detach()

    def filter_str(self, alias, column_rename=False):
        filters = self.filters.get(alias, None)
        if not filters:
            return None
        if not column_rename:
            return ' AND '.join(map(str, filters))
        res = []
        for filter in filters:
            column_name = filter.left_column_name
            new_name = f'_{alias}_{column_name}'
            res.append(filter.to_str(left_rename=new_name))
        return ' AND '.join(res)

    def filter_columns(self, alias):
        filters = self.filters.get(alias, None)
        if not filters:
            return None
        res = set()
        for filter in filters:
            res.add(filter.left_column_name)
        return res

    def _hidden_edge_parent(self, hidden_edge):
        temp = hidden_edge
        parent = self.hidden_edge_union_find_set.setdefault(hidden_edge, hidden_edge)
        while temp != parent:
            temp = parent
            parent = self.hidden_edge_union_find_set.setdefault(parent, parent)
        temp = hidden_edge
        while temp != parent:
            self.hidden_edge_union_find_set[temp] = parent
            temp = self.hidden_edge_union_find_set.get(temp)
        return parent

    def to_hetero_graph_dgl(self, clone=False):
        if not clone:
            g, _data_dict, node_indexes = self.__hetero_graph_dgl
            g = g.to(self.device)
            data_dict = {k : (v[0].to(self.device), v[1].to(self.device)) for k, v in _data_dict.items()}
            return g, data_dict, node_indexes

        node_indexes = {}

        table_nodes_temp = set()
        table_to_table_temp = []

        for left_alias, right_alias, cmp in self.edge_list:
            table_nodes_temp.add(left_alias)
            table_nodes_temp.add(right_alias)

            table_to_table_temp.append((left_alias, right_alias))
            table_to_table_temp.append((right_alias, left_alias))

        table_filters = []
        table_filter_masks = []
        table_edge_features = []
        table_globals = []
        table_onehots = []
        table_others = []

        for index, alias in enumerate(table_nodes_temp):
            node_indexes['~' + alias] = index

            table_filters.append(self.table_feature_filter[alias])
            table_filter_masks.append(self.table_feature_filter_mask[alias])
            table_edge_features.append(self.table_feature_edge[alias])
            table_globals.append(self.table_feature_global[alias])
            table_onehots.append(self.table_feature_onehot[alias])
            table_others.append(self.table_feature_others[alias])

        table_to_table_edge_temp = []
        for left_alias, right_alias in table_to_table_temp:
            table_to_table_edge_temp.append([node_indexes['~' + left_alias], node_indexes['~' + right_alias]])

        data_dict = {}
        x_dict = {
            'table_filter': torch.cat(table_filters, dim=0),
            'table_filter_mask': torch.cat(table_filter_masks, dim=0),
            'table_edge': torch.cat(table_edge_features, dim=0), 'table_global': torch.cat(table_globals, dim=0),
            'table_onehot': torch.stack(table_onehots, dim=0),
            'table_others': torch.cat(table_others, dim=0),
        }

        edge = torch.tensor(table_to_table_edge_temp, device=self.device, dtype=torch.long)
        if torch.numel(edge) == 0:
            edge = torch.zeros(0, 2, device=self.device, dtype=torch.long)
        data_dict['table', 'to', 'table'] = tuple(edge.t())

        g = dgl.heterograph(data_dict, device=self.device)

        g.nodes['table'].data['filter'] = x_dict['table_filter']
        g.nodes['table'].data['filter_mask'] = x_dict['table_filter_mask']
        g.nodes['table'].data['edge'] = x_dict['table_edge']
        g.nodes['table'].data['global'] = x_dict['table_global']
        g.nodes['table'].data['onehot'] = x_dict['table_onehot']
        g.nodes['table'].data['others'] = x_dict['table_others']

        return g, data_dict, node_indexes

    def __column_cmp_edge_type(self, cmp: Comparison):
        op = cmp.op
        if op == '=':
            return (0, 0)
        elif op == '<':
            return (1, 2)
        elif op == '>':
            return (2, 1)
        elif op == '<=':
            return (3, 4)
        elif op == '>=':
            return (4, 3)
        return (0, 0)

    def __column_cmp_edge_str(self, cmp: Comparison):
        l, r = self.__column_cmp_edge_type(cmp)
        return f'op{l}', f'op{r}' # op0 - op4

    @classmethod
    def __onehot(cls, cmp : Comparison, device=torch.device('cpu'), rev=False):
        op = cmp.op
        res = torch.zeros(database.config.edge_onehot, device=device)
        if rev:
            if op[0] == '>':
                op = '<' + op[1:]
            elif op[0] == '<':
                op = '>' + op[1:]
        if op in ('=', '>=', '<='):
            res[0] = 1
        if op in ('>', '>='):
            res[1] = 1
        if op in ('<', '<='):
            res[2] = 1
        return res.unsqueeze(0)

    def get_feature_index(self, column_index, offset):
        return column_index * self.feature_length + offset

    def get_table(self, alias):
        table_name = str(self.alias_to_table[alias])
        table_index, table = database.schema.table(self.alias_to_table[alias].fullname)
        return table_index, table_name, table

    def cost(self):
        for pre in self.pre_actions:
            database.result(pre)
        res = database.cost(self.sql)
        for post in self.post_actions:
            database.result(post)
        return res

    def latency(self):
        for pre in self.pre_actions:
            database.result(pre)
        res = database.latency(self.sql)
        for post in self.post_actions:
            database.result(post)
        return res


class Baseline:
    __re_alias_in_condition = re.compile(r'[( ]([A-Za-z0-9_]+)\.')
    __re_and = re.compile(r'\s[Aa][Nn][Dd]\s')

    def __init__(self, sql):
        if isinstance(sql, dict):
            self.sql = None
            self.plan = sql
        else:
            self.sql = str(sql)
            if isinstance(sql, Sql) and sql.related_subquery:
                self.plan = None
            else:
                self.plan = database.plan(self.sql, geqo=False)[0][0][0]['Plan']
        self.parent = {}
        self.children = {}
        self.attributes = {}
        self.aliases = set()
        self.left_deep = True
        self.join_types = {}

        self.join_counts = {}

        self.__node_index = 0
        self.__joined = set()
        self.join_candidates = []
        self.root = None

        self.__join_order_cache = None
        self.__join_order_table_cache = None
        self.__join_method_cache = None

        if self.plan:
            self.__parse(self.plan)

    @property
    def join_order(self):
        if self.__join_order_cache is not None:
            return list(self.__join_order_cache)
        nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            children = self.children.get(node, None)
            if children:
                nodes.append((node, *children))
                stack.extend(children)
        nodes.reverse()
        dic = {}
        res = []
        for node, left, right in nodes:
            dic[node] = len(dic)
            left = dic.get(left, left)
            right = dic.get(right, right)
            res.append((left, right))
        self.__join_order_cache = res
        return res

    @property
    def join_methods(self):
        if self.__join_method_cache is not None:
            return list(self.__join_method_cache)
        nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            children = self.children.get(node, None)
            if children:
                nodes.append(node)
                stack.extend(children)
        nodes.reverse()
        res = []
        for node in nodes:
            res.append(self.join_types[node])
        self.__join_method_cache = res
        return res

    def __is_parent(self, child, parent):
        _parent = child
        while _parent is not None:
            if _parent == parent:
                return True
            _parent = self.parent.get(_parent, None)
        return False

    @property
    def _result_order(self):
        if self.__join_order_table_cache is not None:
            return list(self.__join_order_table_cache)
        nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            children = self.children.get(node, None)
            if children:
                nodes.append((node, *children))
                stack.extend(children)
        nodes.reverse()
        joined = set(self.join_candidates)
        res = []
        for node, left, right in nodes:
            for left_alias, right_alias in joined:
                if self.__is_parent(left_alias, left) and self.__is_parent(right_alias, right):
                    res.append((left_alias, right_alias))
                    joined.remove((left_alias, right_alias))
                    break
                elif self.__is_parent(right_alias, left) and self.__is_parent(left_alias, right):
                    res.append((right_alias, left_alias))
                    joined.remove((left_alias, right_alias))
                    break
            else:
                # assert False, (nodes, self.join_candidates, (node, left, right))
                break
        if len(res) != len(nodes):
            res = []
        self.__join_order_table_cache = res
        return res

    @property
    def result_order(self):
        return self.join_order

    def __add_condition(self, cond, alias=None):
        if re.search(self.__re_and, cond): # ((t.id = mc.movie_id) AND (it.id = mi_idx.info_type_id))
            conds = re.split(self.__re_and, cond)
            self.__add_condition(conds[0][1:])
            self.__add_condition(conds[1])
            return
        join = [] if alias is None else [alias]
        join.extend(re.findall(self.__re_alias_in_condition, cond))
        if len(join) != 2:
            # the condition might be a filter, e.g.: (info_type_id = 101)
            return
        left, right = join[0], join[1]
        left_in, right_in = left in self.__joined, right in self.__joined

        self.join_counts[left] = self.join_counts.get(left, 0) + 1
        self.join_counts[right] = self.join_counts.get(right, 0) + 1

        if self.__joined and not left_in and not right_in:
            self.left_deep = False ##
        if not (left_in and right_in):
            if left_in:
                _jc = (left, right)
            else:
                _jc = (right, left)
            self.join_candidates.append(_jc)
            self.__joined.update(_jc)

    def __str(self, node, indent=0):
        if node is None:
            return "()"
        children = self.children.get(node, None)
        if children is None:
            return str(node)
        left, right = self.__str(children[0], indent + 1), self.__str(children[1], indent + 1)
        return f"({node}, {left}, {right})"

    def __str__(self):
        return self.__str(self.root)

    def __attributes(self, plan):
        if 'Actual Rows' in plan:
            actual = plan['Actual Rows'] * plan['Actual Loops']
        else:
            actual = None
        return (
            plan['Plan Rows'],
            plan['Plan Width'],
            actual,
        )

    def __parse(self, plan, parent=None, _right=False, _attrs=None):
        node_type = plan['Node Type']
        attrs = self.__attributes(plan) if _attrs is None else _attrs
        if node_type[-4:] == 'Scan':
            # leaf
            alias = plan['Alias']

            assert alias not in self.aliases


            self.aliases.add(alias)
            self.parent[alias] = parent
            self.attributes[alias] = attrs
            if parent is not None:
                children = self.children.setdefault(parent, [])
                children.append(alias)
            else:
                self.root = alias
        elif node_type in ('Nested Loop', 'Merge Join', 'Hash Join'):
            # branch
            plans = plan['Plans']
            if _right:
                self.left_deep = False

            left, right = plans
            node_index = self.__node_index
            if node_type == 'Nested Loop':
                self.join_types[node_index] = 0
            elif node_type == 'Hash Join':
                self.join_types[node_index] = 1
            elif node_type == 'Merge Join':
                self.join_types[node_index] = 2
            else:
                self.join_types[node_index] = -1

            self.parent[node_index] = parent
            self.attributes[node_index] = attrs
            if parent is not None:
                children = self.children.setdefault(parent, [])
                children.append(node_index)
            else:
                self.root = node_index

            self.__node_index += 1
            self.__parse(left, parent=node_index)
            self.__parse(right, parent=node_index, _right=True)

        else:
            # others
            plans = plan['Plans']

            self.__parse(plans[0], parent=parent, _right=_right, _attrs=attrs)

        conditioned = False
        cond = plan.get('Recheck Cond', None)
        if cond is not None:
            alias = plan.get('Alias', None)
            if alias is not None and cond.find('=') >= 0:
                conditioned = True
                self.__add_condition(cond, alias)
        cond = plan.get('Index Cond', None)
        if cond is not None:
            alias = plan.get('Alias', None)
            if alias is not None and cond.find('=') >= 0:
                conditioned = True
                self.__add_condition(cond, alias)
        cond = plan.get('Hash Cond', None)
        if cond is not None:
            conditioned = True
            self.__add_condition(cond)
        cond = plan.get('Merge Cond', None)
        if cond is not None:
            conditioned = True
            self.__add_condition(cond)
        cond = plan.get('Join Filter', None)
        if cond is not None:
            conditioned = True
            self.__add_condition(cond)

    def mapping(self, plan):
        base_to_plan = {a : a for a in plan.sql.aliases}
        plan_to_base = dict(base_to_plan)
        def traverse(node, visited):
            visited.add(node)
            parent = self.parent.get(node, None)
            if parent is None:
                return
            plan_node = base_to_plan.get(node, None)
            if plan_node is not None:
                plan_parent = plan.direct_parent.get(plan_node, None)
                if plan_parent is not None:
                    base_to_plan[parent] = plan_parent
                    plan_to_base[plan_parent] = parent
            traverse(parent, visited=visited)
        visited = set()
        for alias in self.aliases:
            traverse(alias, visited)
        return base_to_plan, plan_to_base

    def mapping_attrs(self, plan):
        base_to_plan, plan_to_base = self.mapping(plan)
        res = {}
        for alias in plan.sql.aliases:
            res[alias] = self.attributes[alias]
        for alias in range(plan.total_branch_nodes):
            res[alias] = self.attributes[plan_to_base[alias]]
        return res

    @classmethod
    def candidate_steps(cls, plan, bushy=False):
        node_map = {a : a for a in plan.sql.aliases}
        baseline = cls(str(plan))
        if not bushy and not baseline.left_deep:
            return []
        def _topo(node, visited):
            visited.add(node)
            parent = baseline.parent.get(node, None)
            if parent is None:
                return [node]
            plan_node = node_map.get(node, None)
            if plan_node is not None:
                plan_parent = plan.direct_parent.get(plan_node, None)
                if plan_parent is not None:
                    node_map[parent] = plan_parent
            res = _topo(parent, visited=visited)
            if parent in visited:
                return [node]
            res.append(node)
            return res
        def traverse(node, record):
            if not isinstance(node, int):
                record.append(node)
                return
            l, r = baseline.children[node]
            traverse(l, record)
            traverse(r, record)
        visited = set()
        topo = []
        for alias in baseline.aliases:
            topo.extend(_topo(alias, visited))
        candidates = set()
        _candidates_set = set()
        ancestors = {}
        for baseline_node in reversed(topo):
            if not baseline_node in node_map:
                children = baseline.children.get(baseline_node, None)
                if children is not None:
                    left, right = children
                    if left in node_map and right in node_map:
                        if bushy or plan.total_branch_nodes == 0 or isinstance(left, int) and not isinstance(right, int):
                            candidates.add((left, right))
                            _candidates_set.add(left)
                            _candidates_set.add(right)
                        if bushy or plan.total_branch_nodes == 0 or isinstance(right, int) and not isinstance(left, int):
                            candidates.add((right, left))
                            _candidates_set.add(left)
                            _candidates_set.add(right)
        #return candidates
        for c in _candidates_set:
            record = []
            traverse(c, record)
            for child in record:
                ancestors[child] = c
        res = []
        for left, right in baseline.join_candidates:
            aleft, aright = ancestors.get(left, None), ancestors.get(right, None)
            if (aleft, aright) in candidates:
                res.append((left, right))
            if (aright, aleft) in candidates:
                res.append((right, left))
        if False and len(candidates) == 0:
            print(str(plan))
        return res
