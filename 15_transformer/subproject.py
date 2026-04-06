import os
from io import BytesIO
import sqlite3
import numpy as np
import pandas as pd

load_meta = lambda db_con: next(pd.read_sql('SELECT * FROM meta LIMIT 1', con=db_con).itertuples())

load_vocab_tokens = lambda db_con: pd.read_sql('SELECT * FROM vocab_tokens', con=db_con)

load_pos_tokens = lambda db_con: pd.read_sql('SELECT * FROM pos_tokens', con=db_con)

def load_image(image_ind, db_con):
    data = db_con.cursor().execute('SELECT data FROM images WHERE image_ind=:image_ind', dict(image_ind=int(image_ind))).fetchone()[0]

    with BytesIO(data) as b:
        return np.load(b)