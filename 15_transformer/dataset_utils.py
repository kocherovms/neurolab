import os
from io import BytesIO
import sqlite3
import numpy as np
import pandas as pd

from utils import *

load_meta = lambda db_con: next(pd.read_sql('SELECT * FROM meta LIMIT 1', con=db_con).itertuples())

load_vocab_tokens = lambda db_con: pd.read_sql('SELECT * FROM vocab_tokens', con=db_con, index_col='token_ind')

load_pos_tokens = lambda db_con: pd.read_sql('SELECT * FROM pos_tokens', con=db_con, index_col='token_ind')

def load_image(db_con, image_ind, is_test=False, with_label=False):
    table_name = ('', 'test_')[is_test] + 'images'
    data, label = db_con.cursor().execute(f'SELECT data, label FROM {table_name} WHERE image_ind=:image_ind', dict(image_ind=int(image_ind))).fetchone()

    with BytesIO(data) as b:
        data = np.load(b)

    return LangUtils.when(with_label, (data, label), data)