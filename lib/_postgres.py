try:
    import psycopg2 as pg
    from psycopg2._psycopg import connection as Connection, cursor as Cursor
except ImportError:
    from sys import stderr
    print('Error: cannot import psycopg2.', file=stderr)
    class Cursor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def fetchall(self):
            return []

        def fetchone(self):
            return ()

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def cursor(self):
            return

    class pg:
        @classmethod
        def connect(cls, *args, **kwargs):
            return Connection()
