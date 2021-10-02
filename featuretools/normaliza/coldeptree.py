from pandas import DataFrame

from .coldepset import ColDepSet
from .coldepnode import ColDepNode


class ColDepTree:
    def __init__(self, df: DataFrame):
        self._roots = set()
        self._build(ColDepSet(df))

    def _build(self, coldeps: ColDepSet):
        for coldep in coldeps:
            colnames = coldep.get_lhs_cols()
            node_lhs = self._find_node(colnames)
            if node_lhs is None:
                node_lhs = ColDepNode(colnames)
                self._roots.add(node_lhs)

            colnames = coldep.get_rhs_col()
            node_rhs = self._find_node_from_subtree(node_lhs.get_root(), colnames)
            if node_rhs is None:
                node_rhs = ColDepNode(colnames)
                node_lhs.add_child(node_rhs)
            elif not node_lhs.is_ancestor(node_rhs):
                if node_lhs.get_level() >= node_rhs.get_level():
                    node_rhs.crop()
                    node_lhs.add_child(node_rhs)

    def _find_node(self, colnames: str):
        for root in self._roots:
            node = self._find_node_from_subtree(root, colnames)
            if node is not None:
                return node
        return None

    def _find_node_from_subtree(self, top: ColDepNode, colnames: str):
        if top.is_colnames(colnames):
            return top
        for sib in top:
            top_sib = self._find_node_from_subtree(sib, colnames)
            if top_sib is not None:
                return top_sib
        return None

    def __repr__(self):
        desc = ""
        for root in self._roots:
            desc += str(root)
        return desc
