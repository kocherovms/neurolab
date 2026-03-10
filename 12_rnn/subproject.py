import sqlite3
import numpy as np
import torch
import torch.utils.data
import re
from collections import Counter
from enum import Flag, StrEnum, auto 
import pandas as pd

class TokenLevel(StrEnum):
    SYMBOL = auto()
    WORD = auto()

class TextPreprocessor:
    def __init__(self, token_level):
        match token_level:
            case TokenLevel.SYMBOL:
                self.split_line = self.split_line_symbol
                self.check_token = self.check_token_symbol
            case TokenLevel.WORD:
                self.split_line = self.split_line_word
                self.check_token = self.check_token_word
            case _:
                assert False, f'Unsupported {token_level=}'
        
    def remove_pg_envelope(self, text):
        start_indx = text.find('START OF THIS PROJECT')
        start_indx = text.find('START OF THE PROJECT') if start_indx < 0 else start_indx
        end_indx = text.find('End of the Project Gutenberg')
        end_indx = text.find('END OF THE PROJECT GUTENBERG') if end_indx < 0 else end_indx
        assert start_indx >= 0
        assert end_indx >= 0
        
        while text[start_indx] != '\n': 
            start_indx += 1
        
        while text[start_indx] == '\n':
            start_indx += 1
        
        while text[end_indx] != '\n':
            end_indx -= 1

        return text[start_indx:end_indx]

    SPLIT_CHARS = r'"\',.:;?!()\[\]=\+\*/—%…\-\$£_&'
    SYMS_SUBST_TAB = {
        'ï': 'i', 'é': 'e', 'ô': 'o', 'î': 'i', 'ê': 'e', 'æ': 'e', 'ä': 'a', 'ç': 'c', 'ö': 'o', 'à': 'a', 'ü': 'u', 'è': 'e',
        'œ': 'e', 'ù': 'u', 'ò': 'o', 'ā': 'a', 'á': 'a', 'ñ': 'n', 'ó': 'o', 'ú': 'u', 'û': 'u', '✠': '', '†': 'd.', 'ë': 'e', 'ſ': 's', '•': '-', '°': '', chr(0x338): '',
        
    }
    
    def preprocess(self, text):
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r'[‘’]', "'", text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)
        text = re.sub(r'\n(\n)+', '\n', text)
        text = re.sub(r'-(-)+', '. ', text)
        text = re.sub(r'\.\.\.', '…', text)
        text = text.lower()
        text = ''.join(map(lambda sym: TextPreprocessor.SYMS_SUBST_TAB.get(sym, sym), text))
        uniq_syms = Counter(text)
        unsupported_syms = list(filter(lambda sym: not re.match(r'[a-z0-9\s' + TextPreprocessor.SPLIT_CHARS + ']', sym), uniq_syms.keys()))
        assert not unsupported_syms, f'Unsupported syms={unsupported_syms}'
        return text

    
    def split_line_symbol(self, line):
        line = re.sub(r'\s(\s)+', ' ', line)
        return list(filter(self.check_token_symbol, line))
        
    def split_line_word(self, line):
        r = []

        # Combination of line.split() and re.split() is used to retain SPLIT_CHARS as individual tokens but to remove separators (\s)
        # Note: re.split with grouping is a must - in opposite case separators will get lost
        for x in line.split():
            r.extend(filter(None, re.split(r'([' + TextPreprocessor.SPLIT_CHARS + r'])', x))) # filter is used to remove empty elements

        return r

    def check_token_word(self, token):
        return re.match(r'[\w' + TextPreprocessor.SPLIT_CHARS +  r']+', token)

    def check_token_symbol(self, token):
        return re.match(r'[\w ' + TextPreprocessor.SPLIT_CHARS +  r']+', token)

class ChunkDataset(torch.utils.data.IterableDataset):
    def __init__(self, db_fname, prefetch_buffer_size=100, max_chunks_count=None, rng=None):
        super().__init__()
        self.db_con = sqlite3.connect(f'file:{db_fname}?mode=ro', uri=True)
        self.vocab_size = self.db_con.execute('SELECT COUNT(rowid) FROM vocab').fetchone()[0]
        self.min_row_id, self.max_row_id = self.db_con.execute('SELECT MIN(rowid), MAX(rowid) FROM chunks').fetchone()
        self.rows_count = self.max_row_id - self.min_row_id + 1
        rows_count_from_db = self.db_con.execute('SELECT COUNT(rowid) FROM chunks').fetchone()[0]
        assert self.rows_count == rows_count_from_db, (self.rows_count, rows_count_from_db, self.min_row_id, self.max_row_id)
        assert max_chunks_count is None or max_chunks_count >= 1
        self.max_chunks_count = max_chunks_count
        self.row_ids = []
        self.row_ids_ind = -1
        self.prefetch_buffer_size = prefetch_buffer_size
        self.prefetch_buffer = None
        self.rng = np.random.default_rng() if rng is None else rng

    def __len__(self):
        return self.rows_count if self.max_chunks_count is None else min(self.rows_count, self.max_chunks_count)
        
    def __iter__(self):
        rows_count_for_iteration = len(self)
        self.row_ids = self.min_row_id + self.rng.choice(self.rows_count, size=rows_count_for_iteration, replace=False, shuffle=False)
        assert len(self.row_ids) == rows_count_for_iteration
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

def load_vocab(db_fname):
    with sqlite3.connect(f'file:{db_fname}?mode=ro', uri=True) as db_con:
        return pd.read_sql('SELECT * FROM vocab', db_con)
