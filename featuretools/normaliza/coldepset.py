import itertools

from pandas import DataFrame

from .rowset_mgr import RowSetManager
from .rowset import RowSet
from .coldep import ColDep


class ColDepSet:
    """
    Column Dependency Set: Group of column dependencies
    """
    def __init__(self, df: DataFrame = None, single_dep: bool = False):
        self._rsm: RowSetManager = RowSetManager(df)
        self._coldeps = set()

        if df is not None:
            self._analyze_column_deps(df, single_dep)

    def add(self, coldep: ColDep):
        self._coldeps.add(coldep)

    def _analyze_column_deps(self, df: DataFrame, single_dep: bool):
        for col in df.columns:
            cols_lhs_cand = set(df.columns)
            cols_lhs_cand.remove(col)
            cnset_rs = frozenset({col})
            rs: RowSet = self._rsm.get(cnset_rs)
            for n in range(1, len(cols_lhs_cand)):
                for lhs in itertools.combinations(cols_lhs_cand, n):
                    if self._is_skip_ok(lhs, rs.cnset):
                        continue
                    cnset_lhs = frozenset(lhs)
                    rs_lhs = self._rsm.get(cnset_lhs)
                    if rs.has_dep(rs_lhs):
                        dep = ColDep(cnset_lhs, cnset_rs)
                        self._coldeps.add(dep)
                if single_dep:
                    break

    def _is_skip_ok(self, cols_lhs, col_rhs):
        for coldep in self._coldeps:
            if coldep.is_wide_dep(cols_lhs, col_rhs):
                return True
        return False

    def __iter__(self):
        return self._coldeps.__iter__()

    def __repr__(self):
        descs = []
        for coldep in self._coldeps:
            descs.append(str(coldep))
        return "\n".join(descs)
