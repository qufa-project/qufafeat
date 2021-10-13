from .rowset import RowSet


class ColDep:
    def __init__(self, cnset_lhs: frozenset, cnset_rhs: frozenset):
        self._cnset_lhs = cnset_lhs
        self._cnset_rhs = cnset_rhs

    def __repr__(self):
        desc_cnset_lhs = "(" + ",".join(self._cnset_lhs) + ")"
        desc_cnset_rhs = "(" + ",".join(self._cnset_rhs) + ")"
        return desc_cnset_lhs + ' -> ' + desc_cnset_rhs

    def get_lhs_cnset(self):
        return self._cnset_lhs

    def get_rhs_cnset(self):
        return self._cnset_rhs

    def is_wide_dep(self, cnset_lhs: frozenset, cnset_rhs: frozenset):
        if self._cnset_lhs.issubset(cnset_lhs) and self._cnset_rhs == cnset_rhs:
            return True
        return False
