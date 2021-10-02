class ColDepNode:
    def __init__(self, colnames: str):
        self._colnames = colnames
        self._childs = []
        self._parent = None
        self._level = 0

    def is_colnames(self, colnames: str):
        if self._colnames == colnames:
            return True
        return False

    def get_level(self):
        level = 1
        parent = self._parent
        while parent is not None:
            level += 1
            parent = parent.get_parent()
        return level

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

    def get_root(self):
        node = self
        while node._parent is not None:
            node = node.get_parent()
        return node

    def is_ancestor(self, ancestor):
        parent = self._parent
        while parent is not None:
            if ancestor == parent:
                return True
            parent = parent.get_parent()
        return False

    def __iter__(self):
        return self._childs.__iter__()

    def __repr__(self):
        desc = "(" + ",".join(self._colnames) + ")\n"
        for child in self._childs:
            for desc_child in str(child).split("\n"):
                if len(desc_child) > 0:
                    desc += (" " + desc_child + "\n")
        return desc
