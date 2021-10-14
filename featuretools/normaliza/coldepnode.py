from typing import Set
from typing import Optional

import featuretools.normaliza.coldeplink as coldeplink


class ColDepNode:
    def __init__(self, cnset: Optional[frozenset]):
        self._cnsets = set()
        if cnset is not None:
            self._cnsets.add(cnset)
        self._links_child: Set[coldeplink.ColDepLink] = set()
        self._links_parent: Set[coldeplink.ColDepLink] = set()
        self._level = 0

    def is_vroot(self):
        if self._cnsets:
            return False
        return True

    def is_invalid(self):
        return self._cnsets is None

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

    def add_link_parent(self, link):
        if len(self._links_parent) == 1:
            for link_parent in self._links_parent:
                if not link_parent.cnset_lhs:
                    link_parent.lhs.remove_link(link_parent)
                    break
        self._links_parent.add(link)

    def add_link_child(self, link):
        self._links_child.add(link)

    def add_link(self, link):
        self.add_link_child(link)
        link.rhs.add_link_parent(link)

    def append_link(self, cnset_lhs, cnset_rhs, child):
        link = coldeplink.ColDepLink(self, child, cnset_lhs, cnset_rhs)
        self.add_link(link)

    def remove_link_parent(self, link, force: bool = False):
        if not force or link in self._links_parent:
            self._links_parent.remove(link)

    def remove_link_child(self, link, force: bool = False):
        if not force or link in self._links_child:
            self._links_child.remove(link)

    def remove_link(self, link, force: bool = False):
        self.remove_link_child(link, force)
        link.rhs.remove_link_parent(link, force)

    def set_link(self, link):
        self.remove_link(link, True)
        self.add_link(link)

    def _has_cn(self, cn):
        for cnset in self._cnsets:
            if cn in cnset:
                return True
        return False

    def _has_cnset(self, cnset):
        for cn in cnset:
            if not self._has_cn(cn):
                return False
        return True

    def has_cnsets(self, cnsets):
        for cnset in cnsets:
            if not self._has_cnset(cnset):
                return False
        return True

    def is_subsumed(self, nd):
        return nd.has_cnsets(self._cnsets)

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
        return link in self._links_parent

    def get_depth(self):
        depth = 1
        for link_parent in self._links_parent:
            depth_parent = link_parent.lhs.get_depth()
            if depth_parent > depth:
                depth = depth_parent
        return depth

    def find(self, cnset: frozenset):
        if self.is_cnset(cnset):
            return self
        for link_child in self._links_child:
            found = link_child.rhs.find(cnset)
            if found is not None:
                return found
        return None

    def validate(self, map_cnset: map = None):
        if self.is_invalid():
            return False
        if map_cnset:
            for cnset in self._cnsets:
                if cnset in map_cnset:
                    if map_cnset[cnset] != self:
                        return False
                else:
                    map_cnset[cnset] = self
        for link_parent in self._links_parent:
            if link_parent.is_invalid():
                return False
            if link_parent.rhs != self:
                return False
            if link_parent.cnset_rhs and not self.is_cnset(link_parent.cnset_rhs):
                return False
            if link_parent.cnset_lhs and not link_parent.lhs.is_cnset(link_parent.cnset_lhs):
                return False
        for link_child in self._links_child:
            if link_child.is_invalid():
                return False
            if link_child.lhs != self:
                return False
            if not self.is_cnset(link_child.cnset_lhs):
                return False
            if not link_child.rhs.is_cnset(link_child.cnset_rhs):
                return False
            if not link_child.rhs.has_parent_link(link_child):
                return False
            if not link_child.rhs.validate(map_cnset):
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

    def collapse(self, node):
        node.add_cnset(self._cnsets)
        for link_child in self._links_child:
            if not link_child.rhs.is_invalid() and node is not link_child.rhs:
                link_new = coldeplink.ColDepLink(node, link_child.rhs, link_child.cnset_lhs, link_child.cnset_rhs)
                node.set_link(link_new)

    def squash(self, node):
        if self == node:
            return

        parents = set()
        for link_parent in self._links_parent:
            link_parent.lhs.remove_link_child(link_parent)
            if link_parent.lhs == node or link_parent.lhs.is_ancestor(node):
                parents.add(link_parent.lhs)
            else:
                link_parent.lhs.append_link(link_parent.cnset_lhs, link_parent.cnset_rhs, node)

        for parent in parents:
            if not parent.is_invalid():
                parent.squash(node)

        self.collapse(node)
        # invalidate myself
        self._cnsets = None

    def make_single_parent(self):
        for link_child in self._links_child.copy():
            link_child.rhs.make_single_parent()

        if len(self._links_parent) > 1:
            depth = 0
            link_single = None
            for link_parent in self._links_parent:
                if link_single is None:
                    link_single = link_parent
                    depth = link_parent.lhs.get_depth()
                else:
                    depth_parent = link_parent.lhs.get_depth()
                    if depth_parent > depth:
                        depth = depth_parent
                        link_single = link_parent

            for link_parent in self._links_parent.copy():
                if link_parent != link_single:
                    link_parent.lhs.remove_link(link_parent)

    def subsumes_children(self):
        for link_child in self._links_child.copy():
            link_child.rhs.subsumes_children()
            if link_child.rhs.is_subsumed(self):
                link_child.rhs.squash(self)

    def __iter__(self):
        from .iterchild import IteratorChild
        return IteratorChild(self._links_child)

    def _get_cnsets_desc(self):
        if self._cnsets is None:
            return "(!invalid!)"
        descs = []
        for cnset in self._cnsets:
            descs.append("(" + ",".join(cnset) + ")")
        return "|".join(descs)

    def get_desc(self, traversed: list, recursive: bool):
        desc = self._get_cnsets_desc()
        if self in traversed:
            return "@" + desc
        traversed.append(self)

        if not recursive:
            return desc

        child_descs = []
        for link_child in self._links_child:
            child_descs.append(" " + str(link_child))
            descs_child = link_child.rhs.get_desc(traversed, recursive).split("\n")
            for desc_child in descs_child:
                child_descs.append("  " + desc_child)
        if len(child_descs) == 0:
            return desc
        return desc + "\n" + "\n".join(child_descs)

    def __repr__(self):
        return self.get_desc([], False)
