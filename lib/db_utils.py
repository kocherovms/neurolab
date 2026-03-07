import os, io
import math

def get_full_db_file_name(config, db_file_name, with_prefix=True):
    base_path = os.path.dirname(os.path.abspath(config.config_fname))
    return os.path.join(os.path.join(base_path, config.dataset_path), ('', config.db_file_name_prefix)[with_prefix] + db_file_name)

def is_table_exists(db_con, table_name):
    cur = db_con.cursor() 
    return len(cur.execute('SELECT name FROM sqlite_master WHERE type=:type AND name=:table_name', {'type': 'table', 'table_name': table_name}).fetchall()) > 0

def is_table_empty(db_con, table_name):
    cur = db_con.cursor() 
    return len(cur.execute(f'SELECT * FROM {table_name} LIMIT 1').fetchall()) < 1

def drop_table_safe(db_con, tn):
    if is_table_exists(db_con, tn):
        db_con.cursor().execute(f'DROP TABLE {tn}')
        db_con.commit()

def get_column_names(db_con, table_name):
    cur = db_con.cursor() 
    return list(map(lambda row: row[1], cur.execute(f'PRAGMA table_info({table_name})').fetchall()))

def ensure_table_columns(db_con, table_name, column_names):
    cur = db_con.cursor() 
    existing_column_names = set(map(lambda row: row[1], cur.execute(f'PRAGMA table_info({table_name})').fetchall()))
    missing_column_names = set(column_names) - existing_column_names

    if not missing_column_names:
        return

    for column_name in missing_column_names:
        cur.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name}')

    db_con.commit()
