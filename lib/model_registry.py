import os
import datetime
import sqlite3 

from .utils import DBUtils

class ModelRegistry:
    def __init__(self, data_path):
        self.db_con = sqlite3.connect(os.path.join(data_path, 'modreg.db'))

        if not DBUtils.is_table_exists(self.db_con, 'models'):
            self.db_con.execute('CREATE TABLE models(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TIMESTAMP, name TEXT, params_count INTEGER, notebook_name TEXT)')
            
    def register_model(self, model, notebook_name):
        cursor = self.db_con.cursor()
        cursor.execute('INSERT INTO models(timestamp, name, params_count, notebook_name) VALUES(:timestamp, :name, :params_count, :notebook_name)', {
            'timestamp': datetime.datetime.now().isoformat(),
            'name': model.__class__.__name__,
            'params_count': sum([p.numel() for p in model.parameters()]),
            'notebook_name': notebook_name
        })
        self.db_con.commit()
        return cursor.execute('SELECT id FROM models WHERE rowid=:rowid', {'rowid': cursor.lastrowid}).fetchone()[0]