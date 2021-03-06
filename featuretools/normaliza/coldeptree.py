from pandas import DataFrame

from .coldepset import ColDepSet
from .coldepnode import ColDepNode


class ColDepTree:
    def __init__(self):
        self._root: ColDepNode = ColDepNode(None)

    def add_root(self, cnset: frozenset, root: ColDepNode):
        self._root.append_link(frozenset(), cnset, root)

    def build(self, coldeps: ColDepSet):
        for coldep in coldeps:
            cnset_lhs = coldep.get_lhs_cnset()
            node_lhs = self._find_node(cnset_lhs)
            if node_lhs is None:
                node_lhs = ColDepNode(cnset_lhs)
                self.add_root(cnset_lhs, node_lhs)

            cnset_rhs = coldep.get_rhs_cnset()
            node_rhs = self._find_node(cnset_rhs)
            if node_rhs is None:
                node_rhs = ColDepNode(cnset_rhs)
                node_lhs.append_link(cnset_lhs, cnset_rhs, node_rhs)
            elif node_lhs is not node_rhs:
                if node_lhs.is_ancestor(node_rhs) or node_rhs.has_descendent(node_lhs):
                    node_lhs.squash(node_rhs)
                else:
                    node_lhs.append_link(cnset_lhs, cnset_rhs, node_rhs)

    def _find_node(self, cnset: frozenset):
        for root in self._root:
            node = root.find(cnset)
            if node is not None:
                return node
        return None

    def validate(self):
        map_cnset = {}
        for root in self._root:
            if not root.validate(map_cnset):
                return False
        return True

    def get_count(self, cnset: frozenset):
        count = 0
        found = set()
        for root in self._root:
            count += root.get_count(cnset, found)
        return count

    def collapse_roots(self):
        root_main = None
        root_squashed = []
        for root in self._root:
            if root_main is None:
                root_main = root
            else:
                root_squashed.append(root)
        for root in root_squashed:
            root.collapse(root_main)
        if root_main:
            self._root = root_main

    def make_single_parent(self):
        """
            make tree where all nodes have a single parent. A shorter path will be removed.
        """
        if self._root.is_vroot():
            for root in self._root:
                root.make_single_parent()
        else:
            self._root.make_single_parent()

    def make_single_child(self):
        """
            make tree where all nodes have a single child.
        """
        if self._root.is_vroot():
            for root in self._root:
                root.make_single_child()
        else:
            self._root.make_single_child()

    def subsumes_children(self):
        """
            simplify tree by reducing the child node which is subsumed by a parent node
        """
        if self._root.is_vroot():
            for root in self._root:
                root.subsumes_children()
        else:
            self._root.subsumes_children()

    def get_norminfos(self):
        norminfos = []
        for child in self._root:
            child.get_norminfos(0, norminfos)
        return norminfos

    def __repr__(self):
        traversed = []

        if self._root.is_vroot():
            root_descs = []
            for root in self._root:
                root_descs.append(root.get_desc(traversed, True))
            return "\n".join(root_descs)
        else:
            return self._root.get_desc(traversed, True)
