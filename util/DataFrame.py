#!/usr/bin/env python
import copy
import numpy as np
from ReadWrite import read_csv
from ReadWrite import read_ws
from PrettyPrint import array_str

class DataFrame():
    """Simple data frame that keeps track of row and column headers"""
    def __init__(self, data, col_names=True, row_names=True, as_factors=True,\
                 none_as=None, val_type=None):
        """Initializes a DataFrame instance for a 2D dataset.
        if col_names and/or row_names is False, arange(n) will be used in place
        for the missing col_names and/or row_names.

        Key Arguments:
        data -- 2D numpy array, list of list, or link to data.
                for csv link, value of |data| is the string of link.
                for xlsx link, value of |data| should be a 
                (workbook_link, worksheet_link) tuple.
        none_as -- value that replaces None values in source data.
        val_type -- dtype of the ndarray of entries. 
                    If val_type is None, dtype will be 'objects'
        as_factors -- if True, set any column with string data to factors.
                      Factor labels begin from 0, numbered according to
                      lexiconic order of the strings
        """
        self._validate_data(data)
        self.none_val = none_as
        self.factor_map = {}
        if val_type is None:
            self.dtype = 'object'
        else:
            self.dtype = val_type
            
        self._data, self._c, self._r = self._construct_data(data, col_names, row_names, none_as, as_factors)
        self.nrows, self.ncols = self._data.shape

    def get_col_index(self, col_name):
        return np.where(self._c == col_name)[0][0]

    def get_entries(self, cpy=True):
        if cpy:
            return copy.deepcopy(self._data)
        else:
            return self._data
    def get_val(self, r, c):
        if isinstance(c, str):
            c = self.get_col_index(c)
        return self._data[r,c]

    def get_col_names(self):
        return copy.deepcopy(self._c)

    def get_row_names(self):
        return copy.deepcopy(self._r)

    def __setitem__(self, key, item):
        self._data[key] = item

    def __getitem__(self, key):
        """overloads __getitem__ operation.
        If key is a single string/number/slice, assume rows are queried.
        Otherwise, key[0] queries rows, key[1] queries cols.
        Returns a new DataFrame instance with the projection.
        """

        key = self._process_key(key)
        entries = np.vstack((self._c.reshape((1,self.ncols))[:,key[1]],self._data[key]))
        new_rows = np.vstack(([0],self._r.reshape((self.nrows,1))[key[0],:]))
        entries = np.hstack((new_rows,entries))
        return DataFrame(entries)
            
    def _process_key(self, key):
        if isinstance(key, tuple):
            return (self._get_row_slice(key[0]), self._get_col_slice(key[1]))
        elif isinstance(key, (slice,list,int,str)):
            return (self._get_row_slice(key), slice(None, None, None))

    def _get_row_slice(self, key):
        return self._get_slice(key, range(self.nrows))
    def _get_col_slice(self, key):
        return self._get_slice(key, self._c)

    def _get_slice(self, key, hdr):
        if isinstance(key, slice):
            s1, s2, s3 = key.start, key.stop, key.step
            if isinstance(s1, list):
                return [self._get_index(elem,hdr) for elem in s1]
            if isinstance(s1, int):
                return key
            return slice(self._get_index(s1,hdr), self._get_index(s2,hdr), s3)
        elif isinstance(key, list):
            return [self._get_index(elem,hdr) for elem in key]
        elif isinstance(key, int):
            return slice(key, key+1, None)
        elif isinstance(key, str):
            s_index = self._get_index(key, hdr)
            if s_index is not None:
                e_index = s_index + 1
            else:
                e_index = None
            return slice(s_index, e_index, None)
        raise ValueError("index key must be list, slice, int,or  string," + \
                         "received {} of type {}".format(key, type(key)))

    def _get_index(self, s, hdr):
        if s is None:
            return None
        try:

            new_key = np.where(np.array(hdr) == s)[0][0]
            return new_key
        except:
            raise ValueError("key {} not found".format(s))

    def __str__(self):

        s = "Columns: " +  str(self._c) + "\n" + "Entries:" + "\n"
        ent = np.hstack((self._r.reshape((self._data.shape[0],1)), self._data))
        s += array_str(ent.tolist())
        return s

    def _construct_data(self, data, h, r, null_val, fac):
        """Returns data entries as numpy array.
        Determines whether to use the 0th row and col as headers based on h and r.
        If null_val is not None, None values are replaced with null_val
        """
        if isinstance(data, str):
            entries = read_csv(data) 
        elif isinstance(data, tuple):
            entries = read_ws(data[0], data[1])
        elif isinstance(data, (list, np.ndarray)):
            entries = data
        entries = np.array(entries, dtype=object)
        if h:
            col_names = np.squeeze(np.array(entries[0,int(r):]))
            entries = entries[1:,:]
        else:
            col_names = np.arange(entries.shape[0])
        if r:
            row_names = np.squeeze(np.array(entries[:,0]))
            entries = entries[:,1:]
        else:
            row_names = np.arange(entries.shape[1])

        if fac:
            self.factor_map = self._to_factors(entries)

        if null_val is not None:
            for i in range(entries.shape[0]):
                for j in range(entries.shape[1]):
                    if entries[i,j] is None:
                        entries[i,j] = null_val
        if self.dtype != 'object':
            entries = entries.astype(float)
        return (entries, col_names, row_names)

    def _to_factors(self, entries):
        factor_map = {}
        for c in range(entries.shape[1]):
            is_factor = self._has_str_entries(entries, c)
            if not is_factor: continue
            factor_map[c] = self._to_factors_col(entries, c)
        return factor_map

    def _to_factors_col(self, entries, c):
        factors = set(entries[:,c].tolist())
        factors.discard(None)
        factors = sorted(list(factors))
        col_map = {}
        for i,factor in enumerate(factors):
            col_map[i] = factor
        for r in range(entries.shape[0]):
            if entries[r,c] is None: continue
            entries[r,c] = factors.index(entries[r,c])
        return col_map

    def _has_str_entries(self, entries, c):
        for r in range(entries.shape[0]):
            if entries[r,c] is None: continue
            try:
                tmp = float(entries[r,c])
            except:
                return True
        return False

    def _validate_data(self, data):
        if isinstance(data, np.ndarray):
            if len(data.shape) != 2:
                raise ValueError("data must be in 2D")
        elif isinstance(data, list):
            pass
        elif isinstance(data, str):
            if not data.endswith(".csv"):
                raise ValueError("data link must end in .csv." + \
                        " For xlsx data link, a (wb,ws) tuple is expected")
        elif isinstance(data, tuple):
            if len(data) < 2 or not data[0].endswith(".xlsx"):
                raise ValueError("xlsx links expect a (wb, ws) tuple")
        else:
            raise ValueError("expects data array or data link(s)")



