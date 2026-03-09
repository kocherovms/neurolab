import sqlite3
import numpy as np
import torch
import torch.utils.data

class ChunkDataset(torch.utils.data.IterableDataset):
    def __init__(self, db_fname, prefetch_buffer_size=100, rng=None):
        super().__init__()
        self.db_con = sqlite3.connect(f'file:{db_fname}?mode=ro', uri=True)
        self.vocab_size = self.db_con.execute('SELECT COUNT(rowid) FROM vocab').fetchone()[0]
        self.min_row_id, self.max_row_id = self.db_con.execute('SELECT MIN(rowid), MAX(rowid) FROM chunks').fetchone()
        self.rows_count = self.max_row_id - self.min_row_id + 1
        assert self.rows_count == self.db_con.execute('SELECT COUNT(rowid) FROM chunks').fetchone()[0], (self.rows_count, self.min_row_id, self.max_row_id)
        self.row_ids = []
        self.row_ids_ind = -1
        self.prefetch_buffer_size = prefetch_buffer_size
        self.prefetch_buffer = None
        self.rng = np.random.default_rng() if rng is None else rng

    def __len__(self):
        return self.rows_count
        
    def __iter__(self):
        self.row_ids = self.min_row_id + self.rng.choice(self.rows_count, size=self.rows_count, replace=False, shuffle=False)
        assert len(self.row_ids) == self.rows_count
        self.row_ids_ind = 0
        self._prefetch()
        return self

    def __next__(self):
        try:
            return next(self.prefetch_buffer)
        except StopIteration:
            self._prefetch()
            return next(self.prefetch_buffer)

    def _prefetch(self):
        fetch_row_ids = self.row_ids[self.row_ids_ind:self.row_ids_ind+self.prefetch_buffer_size]

        if len(fetch_row_ids) == 0:
            raise StopIteration()
            
        self.row_ids_ind += len(fetch_row_ids)
        query = 'SELECT chunk FROM chunks WHERE rowid IN (' + ','.join('?' * len(fetch_row_ids)) + ')'
        rows = self.db_con.execute(query, list(map(int, fetch_row_ids))).fetchall()
        rows = np.array(list(map(lambda row: np.fromstring(row[0], dtype=int, sep=','), rows)))
        rows = torch.tensor(rows)
        assert len(rows) == len(fetch_row_ids), (len(rows), len(fetch_row_ids))
        self.prefetch_buffer = iter(rows)
