from pandas import DataFrame
from typing import Set

from .coldepset import ColDepSet
from .coldepnode import ColDepNode


class ColDepTree:
    def __init__(self, df: DataFrame = None, single_dep: bool = False):
        self._roots: Set[ColDepNode] = set()
        if df is not None:
            self.build(ColDepSet(df, single_dep))

    def add_root(self, root: ColDepNode):
        self._roots.add(root)

    def build(self, coldeps: ColDepSet):
        for coldep in coldeps:
            cnset_lhs = coldep.get_lhs_cnset()
            node_lhs = self._find_node(cnset_lhs)
            if node_lhs is None:
                node_lhs = ColDepNode(cnset_lhs)
                self._roots.add(node_lhs)

            cnset_rhs = coldep.get_rhs_cnset()
            node_rhs = self._find_node(cnset_rhs)
            if node_rhs is None:
                node_rhs = ColDepNode(cnset_rhs)
                node_lhs.set_link(cnset_lhs, cnset_rhs, node_rhs)
            elif node_lhs is not node_rhs:
                if node_lhs.is_ancestor(node_rhs) or node_rhs.has_descendent(node_lhs):
                    node_lhs.squash(node_rhs)
                    if node_lhs in self._roots:
                        self._roots.remove(node_lhs)
                else:
                    if node_rhs in self._roots:
                        self._roots.remove(node_rhs)
                    node_lhs.set_link(cnset_lhs, cnset_rhs, node_rhs)

    def _find_node(self, cnset: frozenset):
        for root in self._roots:
            node = root.find(cnset)
            if node is not None:
                return node
        return None

    def validate(self):
        for root in self._roots:
            if not root.validate():
                return False
        return True

    def get_count(self, cnset: frozenset):
        count = 0
        found = set()
        for root in self._roots:
            count += root.get_count(cnset, found)
        return count

    def collapse_roots(self):
        root_main = self._roots.pop()
        for root in self._roots:
            root.squash(root_main)
        self._roots.clear()
        self._roots.add(root_main)

    def __repr__(self):
        traversed = []
        root_descs = []
        for root in self._roots:
            root_descs.append(root.get_desc(traversed))
        return "\n".join(root_descs)
