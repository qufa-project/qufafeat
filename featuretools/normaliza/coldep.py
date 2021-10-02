from .rowset import RowSet


class ColDep:
    def __init__(self, lhs: RowSet, rhs: RowSet):
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self):
        return self._lhs.get_desc() + ' -> ' + self._rhs.get_desc()

    def get_lhs_cnset(self):
        return self._lhs.cnset

    def get_rhs_cnset(self):
        return self._rhs.cnset

    def is_wide_dep(self, cols_lhs: frozenset, col_rhs: frozenset):
        if self._lhs.cnset.issubset(cols_lhs) and self._rhs.cnset == col_rhs:
            return True
        return False
