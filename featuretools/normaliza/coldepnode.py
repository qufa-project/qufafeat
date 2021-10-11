from typing import Set

import featuretools.normaliza.coldeplink as coldeplink


class ColDepNode:
    def __init__(self, cnset: frozenset):
        self._cnsets = set()
        self._cnsets.add(cnset)
        self._links_child: Set[coldeplink.ColDepLink] = set()
        self._links_parent: Set[coldeplink.ColDepLink] = set()
        self._level = 0

    def is_cnset(self, cnset: frozenset):
        for cnset_my in self._cnsets:
            if cnset_my == cnset:
                return True
        return False

    def add_cnset(self, cnset):
        if isinstance(cnset, frozenset):
            self._cnsets.add(cnset)
        else:
            for c in cnset:
                self._cnsets.add(c)

    def add_parent(self, cnset_lhs, cnset_rhs, parent):
        link_parent = coldeplink.ColDepLink(parent, self, cnset_lhs, cnset_rhs)
        self._links_parent.add(link_parent)

    def add_child(self, cnset_lhs, cnset_rhs, child):
        for link_child in self._links_child:
            if link_child.cnset_lhs == cnset_lhs and link_child.cnset_rhs == cnset_rhs:
                return

        link_child = coldeplink.ColDepLink(self, child, cnset_lhs, cnset_rhs)
        self._links_child.add(link_child)

    def set_link(self, cnset_lhs, cnset_rhs, child):
        self.add_child(cnset_lhs, cnset_rhs, child)
        child.add_parent(cnset_lhs, cnset_rhs, self)

    def remove_parent(self, cnset_lhs, cnset_rhs):
        for link_parent in self._links_parent:
            if link_parent.cnset_lhs == cnset_lhs and link_parent.cnset_rhs == cnset_rhs:
                self._links_parent.remove(link_parent)
                return

    def remove_child(self, cnset_lhs, cnset_rhs):
        for link_child in self._links_child:
            if link_child.cnset_lhs == cnset_lhs and link_child.cnset_rhs == cnset_rhs:
                self._links_child.remove(link_child)
                return

    def has_descendent(self, node) -> bool:
        for link_child in self._links_child:
            if link_child.rhs == node:
                return True
            if link_child.rhs.has_descendent(node):
                return True
        return False

    def is_ancestor(self, ancestor):
        for link_parent in self._links_parent:
            if link_parent.lhs == ancestor:
                return True
            if link_parent.lhs.is_ancestor(ancestor):
                return True
        return False

    def has_parent_link(self, link):
        for link_parent in self._links_parent:
            if link_parent.cnset_lhs == link.cnset_lhs and link_parent.cnset_rhs == link.cnset_rhs and\
                    link_parent.lhs == link.lhs:
                return True
        return False

    def find(self, cnset: frozenset):
        if self.is_cnset(cnset):
            return self
        for link_child in self._links_child:
            found = link_child.rhs.find(cnset)
            if found is not None:
                return found
        return None

    def validate(self):
        for link_parent in self._links_parent:
            if link_parent.rhs != self:
                return False
            if not self.is_cnset(link_parent.cnset_rhs):
                return False
            if not link_parent.lhs.is_cnset(link_parent.cnset_lhs):
                return False
        for link_child in self._links_child:
            if link_child.lhs != self:
                return False
            if not self.is_cnset(link_child.cnset_lhs):
                return False
            if not link_child.rhs.is_cnset(link_child.cnset_rhs):
                return False
            if not link_child.rhs.has_parent_link(link_child):
                return False
            if not link_child.rhs.validate():
                return False
        return True

    def get_count(self, cnset: frozenset, found: set):
        count = 0
        if self not in found and self.is_cnset(cnset):
            count = 1
            found.add(self)
        for link_child in self._links_child:
            count += link_child.rhs.get_count(cnset, found)
        return count

    def _squash_with_node(self, node):
        node.add_cnset(self._cnsets)
        for link_child in self._links_child:
            node.set_link(link_child.cnset_lhs, link_child.cnset_rhs, link_child.rhs)

    def squash(self, node):
        if self == node:
            return

        for link_parent in self._links_parent:
            link_parent.lhs.remove_child(link_parent.cnset_lhs, link_parent.cnset_rhs)

        for link_parent in self._links_parent:
            if link_parent.lhs == node or link_parent.lhs.is_ancestor(node):
                link_parent.lhs.squash(node)
            else:
                link_parent.lhs.set_link(link_parent.cnset_lhs, link_parent.cnset_rhs, node)

        self._squash_with_node(node)

    def __iter__(self):
        return self._links_child.__iter__()

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
        for link_child in self:
            child_descs.append(" " + str(link_child))
            descs_child = link_child.rhs.get_desc(traversed).split("\n")
            for desc_child in descs_child:
                child_descs.append("  " + desc_child)
        if len(child_descs) == 0:
            return desc
        return desc + "\n" + "\n".join(child_descs)

    def __repr__(self):
        return self.get_desc([])
