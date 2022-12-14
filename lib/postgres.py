from . import iterator_utils
import pandas as pd
from tqdm import tqdm
from ._postgres import pg, Connection

PG_TABLES = "select * from pg_tables;"
PG_TABLES_NAME_ONLY = "select schemaname, tablename from pg_tables;"
PG_TABLE_STRUCTURE = """
SELECT a.attnum AS index,
       a.attname AS field,
       t.typname AS type,
       a.attlen AS length,
       a.atttypmod AS lengthvar,
       a.attnotnull AS notnull,
       b.description AS description
  FROM pg_class c,
       pg_namespace n,
       pg_attribute a
       LEFT OUTER JOIN pg_description b ON a.attrelid = b.objoid AND a.attnum = b.objsubid,
       pg_type t
 WHERE c.relname = %s
       and a.attnum > 0
       and a.attrelid = c.oid
       and a.atttypid = t.oid
       and n.nspname = %s
       and n.oid = c.relnamespace
 ORDER BY a.attnum
"""
PG_ROW = "select %s from %s %s %s;"
PG_ROW_COUNT = "select count(*) from %s;"
PG_ROW_UNIQUE = "select %s, count(*) from %s group by %s %s %s;"
PG_ROW_UNIQUE_COUNT = "select count(*) from (select %s from %s group by %s) as %s_t;"
PG_STATS_INFO = "select " \
                "inherited, null_frac, avg_width, n_distinct, " \
                "most_common_vals, most_common_freqs, histogram_bounds, " \
                "correlation, most_common_elems, most_common_elem_freqs, " \
                "elem_count_histogram from pg_stats where " \
                "schemaname = %s and tablename = %s and attname = %s;"

def connect(*args, **kwargs) -> Connection:
    return pg.connect(*args, **kwargs)

def set_seed(connection, seed):
    cur = connection.cursor()
    cur.execute('select setseed(%s);', (seed, ))

def pg_row(field, table_name, schema_name=None, random=False, limit=None):
    if schema_name is not None and schema_name != 'public':
        table_name = "%s.%s" % (schema_name, table_name)
    return PG_ROW % (field, table_name,
                     "order by random()" if random else "",
                     "limit %d" % limit if limit is not None else "",
                     )

def pg_row_count(table_name, schema_name=None):
    if schema_name is not None and schema_name != 'public':
        table_name = "%s.%s" % (schema_name, table_name)
    return PG_ROW_COUNT % (table_name)

def pg_row_unique(field, table_name, schema_name=None, random=False, limit=None):
    if schema_name is not None and schema_name != 'public':
        table_name = "%s.%s" % (schema_name, table_name)
    return PG_ROW_UNIQUE % (field, table_name, field,
                            "order by random()" if random else "",
                            "limit %d" % limit if limit is not None else "",
                            )

def pg_row_unique_count(field, table_name, schema_name=None):
    if schema_name is not None and schema_name != 'public':
        table_name_ = "%s.%s" % (schema_name, table_name)
    else:
        table_name_ = table_name
    return PG_ROW_UNIQUE_COUNT % (field, table_name_, field, table_name)

def _iter_columns(connection, table_names=None, typ='varchar', null=False, random=False, limit=None, unique=True):
    tnames = tables(connection)
    table_permit_rows = {}
    if table_names is not None:
        _table_names = []
        for t in table_names:
            if isinstance(t, tuple):
                _table_names.append(t[0])
                rows = table_permit_rows.setdefault(t[0], set())
                rows.add(t[1])
            else:
                _table_names.append(t)
        tnames = filter(lambda x: x[1] in _table_names, tnames)
    for sname, tname in tnames:
        fields = []
        for index, field, _typ, length, lengthvar, notnull, desc in table_structure(connection, tname, schema_name=sname):
            if typ is not None and _typ != typ:
                continue
            fields.append(field)
        rows = table_permit_rows.get(tname, None)
        if rows is not None:
            fields = filter(lambda x: x in rows, fields)
        for f in fields:
            query = pg_row_unique(f, tname, sname, random=random, limit=limit) if unique else pg_row(f, tname, sname, random=random, limit=limit)
            it = iterator_utils.cursor_iter(connection, query)
            for i in it:
                if null or i[0] is not None:
                    yield (sname, tname, *i)

def iter_columns(connection, table_names=None, typ='varchar', null=False, random=False, limit=None, unique=False, sized=False, verbose=False):
    res = _iter_columns(connection, table_names=table_names, typ=typ, null=null, random=random, limit=limit, unique=unique)
    if sized:
        sz = 0
        gen = tables(connection)
        table_permit_rows = {}
        if table_names is not None:
            _table_names = []
            for t in table_names:
                if isinstance(t, tuple):
                    _table_names.append(t[0])
                    rows = table_permit_rows.setdefault(t[0], set())
                    rows.add(t[1])
                else:
                    _table_names.append(t)
            gen = list(filter(lambda x: x[1] in _table_names, gen))
        if verbose:
            gen = tqdm(gen)
        for sname, tname in gen:
            fields = []
            for index, field, _typ, length, lengthvar, notnull, desc in table_structure(connection, tname, schema_name=sname):
                if typ is not None and _typ != typ:
                    continue
                fields.append(field)
            rows = table_permit_rows.get(tname, None)
            if rows is not None:
                fields = filter(lambda x: x in rows, fields)
            for f in fields:
                query = pg_row_unique_count(f, tname, sname) if unique else pg_row_count(tname, sname)
                it = iterator_utils.cursor_iter(connection, query)
                _sz = list(it)[0][0]
                if limit is not None and _sz > limit:
                    _sz = limit
                sz += _sz
        return iterator_utils.SizedWrapper(res, sz)
    return res

def _iter_column_unique(connection, column, table_name, schema_name='public', random=False):
    it = iterator_utils.cursor_iter(connection, pg_row_unique(column, table_name, schema_name, random))
    for i in it:
        yield i

def iter_column_unique(connection, column, table_name, schema_name='public', random=False):
    sz = iterator_utils.cursor_iter(connection, pg_row_unique_count(column, table_name, schema_name))
    sz = list(sz)[0]
    return iterator_utils.SizedWrapper(_iter_column_unique(connection, column, table_name, schema_name, random), sz)

def tables(connection):
    return list(filter(lambda x: x[0] != 'pg_catalog' and x[0] != 'information_schema', iterator_utils.cursor_iter(connection, PG_TABLES_NAME_ONLY)))

def iter_table(connection, table_name, schema_name=None, filter_type=None):
    if schema_name is not None and schema_name != 'public':
        table_name = "%s.%s" % (schema_name, table_name)
    return iterator_utils.cursor_iter(connection, "select * from %s" % (table_name), filter_type=filter_type)

def filter_iter_table(connection, table_name, schema_name=None, filter=None):
    if schema_name is not None and schema_name != 'public':
        table_name = "%s.%s" % (schema_name, table_name)
    if filter is None:
        filter = ''
    else:
        filter = f' where {filter}'
    return iterator_utils.cursor_iter(connection, "select * from %s%s" % (table_name, filter))

def table_structure(connection, table_name, schema_name='public'):
    cur = connection.cursor()
    cur.execute(PG_TABLE_STRUCTURE, (table_name, schema_name))
    res = cur.fetchall()
    cur.close()
    return res

def iter_database(connection, filter_type=None):
    for table in tables(connection):
        sname, tname = table
        for i in iter_table(connection, tname, sname, filter_type=filter_type):
            yield i

def _read_anyarray(value):
    res = []
    elems = []
    inside = False
    escape = False
    quoted = False
    for c in value[1:-1]:
        if escape:
            elems.append(c)
            escape = False
        elif c == '\\':
            escape = True
        elif c == '"':
            inside = not inside
            quoted = True
        elif not inside and c == ',':
            v = ''.join(elems)
            if not quoted and v == 'null':
                res.append(None)
            else:
                res.append(v)
            elems = []
            quoted = False
        else:
            elems.append(c)
    v = ''.join(elems)
    if not quoted and v == 'null':
        res.append(None)
    else:
        res.append(v)
    return res

def pg_stats(connection, table, column, schema='public'):
    cur = connection.cursor()
    cur.execute(PG_STATS_INFO, (schema, table, column))
    res = cur.fetchall()
    cur.close()
    if len(res) < 1:
        return None
    res = {k : v for k, v in zip((
        'inherited', 'null_frac', 'avg_width', 'n_distinct',
        'most_common_vals', 'most_common_freqs', 'histogram_bounds',
        'correlation', 'most_common_elems', 'most_common_elem_freqs',
        'elem_count_histogram',
    ), res[0])}
    for c in ('most_common_vals', 'histogram_bounds', 'most_common_elems'):
        if res[c] is not None:
            res[c] = _read_anyarray(res[c])
    return res

def db_unpack(connection, filter_type=str):
    return iterator_utils.unpack(iter_database(connection, filter_type=filter_type))

def table_summary(connection, table_name, schema_name='public', rows=10):
    columns = list(map(lambda x: "%s : %s" % (x[1], x[2]), table_structure(connection, table_name, schema_name=schema_name)))
    it = iter_table(connection, table_name)
    data = []
    for i in range(rows):
        try:
            row = next(it)
        except StopIteration:
            break
        data.append(row)
    return pd.DataFrame(data, columns=columns)

def all_table_summary(connection, schema_name='public', rows=10):
    res = {}
    for sname, tname in tables(connection):
        if schema_name is not None and sname != schema_name:
            continue
        res[tname] = table_summary(connection, tname, sname, rows=rows)
    return res

def iter_gen(*args, **kwargs):
    assert len(args) > 0
    def iter_gen():
        return args[0](*args[1:], **kwargs)
    return iter_gen
