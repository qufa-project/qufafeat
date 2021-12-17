from featuretools.variable_types.variable import (
    Numeric,
    Datetime
)

import re

from .columnspec import ColumnSpec


_typestr_to_mkfeat = {
    "date": "date",
    "datetime": "date",
    "numeric": "number",
    "ordinal": "number",
    "boolean": "bool",
    "categorical": "class",
}


class FeatureHelper:
    def __init__(self, features, colspec: ColumnSpec):
        self._features = features
        self._colspec = colspec

    def to_array(self):
        arr = []
        for feature in self._features:
            arr.append([self._get_user_friendly_feature_name(feature.get_name()),
                        self._convert_mkfeat_type_string(feature.variable_type)])
        return arr

    def _get_user_friendly_feature_name(self, name):
        names = []
        for token in self._get_tokens(name):
            names.append(self._get_feature_name_with_str(token))
        return ' '.join(names)

    def _get_feature_name_with_str(self, name):
        if self._is_colname_feature_name(name):
            return name
        parsed = self._parse_table_form(name)
        if parsed:
            parsed2 = self._parse_table_form(parsed[1])
            if parsed2:
                return self._get_feature_name_with_str(parsed[1])
            parsed2 = self._parse_operator_form(parsed[1])
            if parsed2:
                return parsed2[0] + "(" + self._get_feature_name_with_str(parsed2[1]) + ")@" + parsed[0]
            return self._remove_tbl_form(parsed[1])
        parsed = self._parse_operator_form(name)
        if parsed:
            return parsed[0] + "(" + self._get_feature_name_with_str(parsed[1]) + ")"
        return self._remove_tbl_form(name)

    def _is_colname_feature_name(self, name):
        if self._colspec.has_colname(name):
            return True
        return False

    def _parse_table_form(self, name):
        matched = re.match(r"^tbl_([^\.]+)_\d+\.(.+)", name)
        if matched:
            if matched.group(1) == 'main' or self._is_colname_feature_name(matched.group(1)):
                return matched.group(1), matched.group(2)
        return None

    @staticmethod
    def _parse_operator_form(name):
        matched = re.match(r"^(\w+)\((.+)\)$", name)
        if matched:
            return matched.group(1), matched.group(2)
        return None

    def _remove_tbl_form(self, name):
        while True:
            matched = re.match(r"^(.*)tbl_\w+_\d+\.(.+)$", name)
            if not matched:
                return name
            name = matched.group(1) + matched.group(2)

    def _get_tokens(self, name):
        tokens = []
        n_rounds = 0
        token = ''
        for char in name:
            if char == '(':
                n_rounds += 1
            elif char == ')':
                n_rounds -= 1
            if n_rounds == 0 and char == ' ':
                if token:
                    tokens.append(token)
                    token = ''
            else:
                token += char
        if token:
            tokens.append(token)
        return tokens

    @staticmethod
    def _convert_mkfeat_type_string(vartype):
        typestr = vartype.type_string
        if typestr in _typestr_to_mkfeat:
            return _typestr_to_mkfeat[typestr]
        return "string"
