from typing import Optional


class ColDepNode:
    def __init__(self, cnset: frozenset):
        self._cnsets = []
        self._cnsets.append(cnset)
        self._childs = []
        self._parent: Optional[ColDepNode] = None
        self._level = 0

    def is_cnset(self, cnset: frozenset):
        for cnset_my in self._cnsets:
            if cnset_my == cnset:
                return True
        return False

    def get_level(self):
        level = 1
        parent = self._parent
        while parent is not None:
            level += 1
            parent = parent.get_parent()
        return level

    def add_cnset(self, cnset):
        if isinstance(cnset, list):
            self._cnsets = self._cnsets + cnset
        else:
            self._cnsets.append(cnset)

    def add_child(self, child):
        self._childs.append(child)
        child._parent = self

    def remove_child(self, child):
        self._childs.remove(child)
        child._parent = None

    def crop(self):
        self._parent.remove_child(self)

    def get_parent(self):
        return self._parent

    def is_ancestor(self, ancestor):
        parent = self._parent
        while parent is not None:
            if ancestor == parent:
                return True
            parent = parent.get_parent()
        return False

    def is_root(self):
        if self._parent is None:
            return True
        return False

    def get_root(self):
        node = self
        while node._parent is not None:
            node = node.get_parent()
        return node

    def find(self, cnset: frozenset):
        if self.is_cnset(cnset):
            return self
        for child in self._childs:
            found = child.find(cnset)
            if found is not None:
                return found
        return None

    def _squash_with_parent(self):
        self._parent.add_cnset(self._cnsets)
        for child in self._childs:
            self._parent.add_child(child)

    def squash(self, ancestor):
        if self == ancestor:
            return
        self._squash_with_parent()
        self._parent.squash(ancestor)

    def __iter__(self):
        return self._childs.__iter__()

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
        for child in self._childs:
            for desc_child in child.get_desc(traversed).split("\n"):
                child_descs.append(" " + desc_child)
        if len(child_descs) == 0:
            return desc
        return desc + "\n" + "\n".join(child_descs)

    def __repr__(self):
        return self.get_desc([])
