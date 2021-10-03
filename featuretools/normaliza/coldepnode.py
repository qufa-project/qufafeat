from typing import Set


class ColDepNode:
    def __init__(self, cnset: frozenset):
        self._cnsets = []
        self._cnsets.append(cnset)
        self._childmap = []
        self._parents: Set[ColDepNode] = set()
        self._level = 0

    def is_cnset(self, cnset: frozenset):
        for cnset_my in self._cnsets:
            if cnset_my == cnset:
                return True
        return False

    def add_cnset(self, cnset):
        if isinstance(cnset, list):
            self._cnsets = self._cnsets + cnset
        else:
            self._cnsets.append(cnset)

    def add_child(self, cnset, child):
        self._childmap.append([cnset, child])
        child._parent = self

    def remove_child(self, child):
        for cinfo in self._childmap:
            if cinfo[1] == child:
                self._childmap.remove(cinfo)
                child._parent = None
                return

    def crop(self):
        for parent in self._parents:
            parent.remove_child(self)

    def get_parents(self):
        return self._parents

    def get_child_lhs_cnset(self, child):
        for cinfo in self._childmap:
            if cinfo[1] == child:
                return cinfo[0]
        return None

    def has_descendent(self, child_cnset) -> bool:
        if self.find(child_cnset) is not None:
            return True
        return False

    def is_ancestor(self, ancestor):
        for parent in self._parents:
            if parent == ancestor:
                return True
            return parent.is_ancestor(ancestor)
        return False

    def find(self, cnset: frozenset):
        if self.is_cnset(cnset):
            return self
        for cinfo in self._childmap:
            found = cinfo[1].find(cnset)
            if found is not None:
                return found
        return None

    def _squash_with_node(self, node):
        node.add_cnset(self._cnsets)
        for cinfo in self._childmap:
            node.add_child(cinfo[0], cinfo[1])

    def squash(self, node):
        if self == node:
            return

        for parent in self._parents:
            if parent == node or parent.is_ancestor(node):
                parent.remove_child(self)
                parent.squash(node)
            else:
                child_lhs_cnset = parent.get_child_lhs_cnset(self)
                parent.remove_child(self)
                if parent not in node.get_parents():
                    parent.add_child(child_lhs_cnset, node)

        self._squash_with_node(node)

    def __iter__(self):
        return self._childmap.__iter__()

    def _get_cnsets_desc(self):
        descs = []
        for cnset in self._cnsets:
            descs.append("(" + ",".join(cnset) + ")")
        return "|".join(descs)

    def get_desc(self, traversed: list):
        desc = self._get_cnsets_desc()
        if self in traversed:
            return "@" + desc
        traversed.append(self)

        child_descs = []
        for cinfo in self._childmap:
            desc_lhs_cnset = "(" + ",".join(cinfo[0]) + ")"
            descs_child = cinfo[1].get_desc(traversed).split("\n")
            child_descs.append(" " + desc_lhs_cnset + "->" + descs_child.pop(0))
            for desc_child in descs_child:
                child_descs.append("  " + desc_child)
        if len(child_descs) == 0:
            return desc
        return desc + "\n" + "\n".join(child_descs)

    def __repr__(self):
        return self.get_desc([])
