from pandas import DataFrame

from .coldepset import ColDepSet
from .coldepnode import ColDepNode


class ColDepTree:
    def __init__(self, df: DataFrame):
        self._roots = set()
        self._build(ColDepSet(df))

    def _build(self, coldeps: ColDepSet):
        for coldep in coldeps:
            cnset = coldep.get_lhs_cnset()
            node_lhs = self._find_node(cnset)
            if node_lhs is None:
                node_lhs = ColDepNode(cnset)
                self._roots.add(node_lhs)

            cnset = coldep.get_rhs_cnset()
            node_rhs = self._find_node(cnset)
            if node_rhs is None:
                node_rhs = ColDepNode(cnset)
                node_lhs.add_child(node_rhs)
            else:
                if node_lhs.is_ancestor(node_rhs):
                    node_lhs.squash(node_rhs)
                else:
                    if node_rhs.is_root():
                        self._roots.remove(node_rhs)
                    node_lhs.add_child(node_rhs)

    def _find_node(self, cnset: frozenset):
        for root in self._roots:
            node = root.find(cnset)
            if node is not None:
                return node
        return None

    def _is_root(self, node):
        if node in self._roots:
            return True
        return False

    def __repr__(self):
        traversed = []
        root_descs = []
        for root in self._roots:
            root_descs.append(root.get_desc(traversed))
        return "\n".join(root_descs)
