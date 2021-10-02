from .rowset import RowSet


class ColDep:
    def __init__(self, lhs: RowSet, rhs: RowSet):
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self):
        return self._lhs.get_desc() + ' -> ' + self._rhs.get_desc()

    def get_lhs_cols(self):
        return self._lhs.colnames

    def get_rhs_col(self):
        return self._rhs.colnames

    def is_wide_dep(self, cols_lhs: frozenset, col_rhs: frozenset):
        if self._lhs.colnames.issubset(cols_lhs) and self._rhs.colnames == col_rhs:
            return True
        return False
