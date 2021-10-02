import itertools

from pandas import DataFrame

from .rowset_mgr import RowSetManager
from .rowset import RowSet
from .coldep import ColDep


class ColDepSet:
    """
    Column Dependency Set: Group of column dependencies
    """
    def __init__(self, df: DataFrame):
        self._rsm: RowSetManager = RowSetManager(df)
        self._coldeps = set()

        self._analyze_column_deps(df)

    def _analyze_column_deps(self, df: DataFrame):
        for col in df.columns:
            cols_lhs_cand = set(df.columns)
            cols_lhs_cand.remove(col)
            rs: RowSet = self._rsm.get(frozenset({col}))
            for n in range(1, len(cols_lhs_cand)):
                for lhs in itertools.combinations(cols_lhs_cand, n):
                    if self._is_skip_ok(lhs, rs.cnset):
                        continue
                    rs_lhs = self._rsm.get(frozenset(lhs))
                    if rs.has_dep(rs_lhs):
                        dep = ColDep(rs_lhs, rs)
                        self._coldeps.add(dep)

    def _is_skip_ok(self, cols_lhs, col_rhs):
        for coldep in self._coldeps:
            if coldep.is_wide_dep(cols_lhs, col_rhs):
                return True
        return False

    def __iter__(self):
        return self._coldeps.__iter__()
