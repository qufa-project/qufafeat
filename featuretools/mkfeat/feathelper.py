from featuretools.variable_types.variable import (
    Numeric,
    Datetime
)


_typestr_to_mkfeat = {
    "boolean": "boolean",
    "categorical": "category",
    "country_code": "string",
    "date": "date",
    "date_of_birth": "datetime",
    "datetime": "datetime",
    "datetime_time_index": "datetime",
    "discrete": "category",
    "email_address": "string",
    "file_path": "string",
    "full_name": "string",
    "id": "numeric",
    "index": "numeric",
    "ip_address": "string",
    "lat_long": "tuple",
    "natural_language": "string",
    "numeric": "numeric",
    "numeric_time_index": "numeric",
    "ordinal": "category",
    "phone_number": "string",
    "sub_region_code": "string",
    "time_index": "datetime",
    "timedelta": "timedelta",
    "url": "string",
    "zip_code": "string"
}


class FeatureHelper:
    def __init__(self, features):
        self._features = features

    def to_array(self):
        arr = []
        for feature in self._features:
            arr.append([feature.get_name(), self._convert_mkfeat_type_string(feature.variable_type)])
        return arr

    @staticmethod
    def _convert_mkfeat_type_string(vartype):
        typestr = vartype.type_string
        if typestr in _typestr_to_mkfeat:
            return _typestr_to_mkfeat[typestr]
        return "string"
