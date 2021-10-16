import itertools

from pandas import DataFrame

from .rowset_mgr import RowSetManager
from .rowset import RowSet
from .coldep import ColDep


class ColDepSet:
    """
    Column Dependency Set: Group of column dependencies
    """
    def __init__(self, single_dep: bool):
        self._coldeps = set()
        self._single_dep = single_dep

    def add(self, coldep: ColDep):
        self._coldeps.add(coldep)

    def analyze(self, df: DataFrame):
        if not self._coldeps:
            self._analyze_coldeps(df)
            return False
        else:
            return self._analyze_with_my_coldeps(df)

    def _analyze_coldeps(self, df: DataFrame):
        rsm: RowSetManager = RowSetManager(df)

        for col in df.columns:
            cols_lhs_cand = set(df.columns)
            cols_lhs_cand.remove(col)
            cnset_rs = frozenset({col})
            rs: RowSet = rsm.get(cnset_rs)
            for n in range(1, len(cols_lhs_cand)):
                for lhs in itertools.combinations(cols_lhs_cand, n):
                    if self._is_skip_ok(lhs, rs.cnset):
                        continue
                    cnset_lhs = frozenset(lhs)
                    rs_lhs = rsm.get(cnset_lhs)
                    if rs.has_dep(rs_lhs):
                        dep = ColDep(cnset_lhs, cnset_rs)
                        self._coldeps.add(dep)
                if self._single_dep:
                    break

    def _analyze_with_my_coldeps(self, df: DataFrame):
        rsm: RowSetManager = RowSetManager(df)

        coldeps_ok = set()
        for coldep in self._coldeps:
            rs_lhs = rsm.get(coldep.get_lhs_cnset())
            rs_rhs = rsm.get(coldep.get_rhs_cnset())
            if rs_rhs.has_dep(rs_lhs):
                coldeps_ok.add(coldep)
        is_stable = False
        if len(coldeps_ok) == len(self._coldeps):
            is_stable = True
        self._coldeps = coldeps_ok
        return is_stable

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
