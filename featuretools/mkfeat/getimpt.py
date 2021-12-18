#!/usr/bin/env python3

import sys
import os
import json
from os.path import dirname

if __package__ is None:
    import logger
else:
    from . import logger


def usage():
    print("""\
Usage: getimpt.py <configuration in json>
""")


def load_conf(path_conf: str):
    with open(path_conf) as f:
        try:
            conf = json.load(f)
        except json.decoder.JSONDecodeError:
            logger.error("configuration has wrong format")
            exit(2)
        if 'data' not in conf:
            logger.error("configuration does not have data")
            exit(2)
        if 'uri' not in conf['data']:
            logger.error("configuration does not have uri in data")
            exit(2)
        if 'columns' not in conf['data'] and 'columns_default' not in conf['data']:
            logger.error("configuration does not have columns in data")
            exit(2)
        if 'label' in conf:
            if 'uri' not in conf['label']:
                logger.error("configuration does not have uri in label")
                exit(2)
            if 'columns' not in conf['label']:
                logger.error("configuration does not have columns in label")
                exit(2)
        return conf


def _get_columninfo_from_columns_default(colname, columns_default):
    for ci in columns_default:
        if ci["name"] == colname:
            return ci
    return None


def _build_columns_from_csv(path_input: str, columns_default: list):
    import pandas as pd
    columns = []

    data = pd.read_csv(path_input, nrows=1)
    type_default = "string"
    ci_def = _get_columninfo_from_columns_default('*', columns_default)
    if ci_def:
        type_default = ci_def['type']

    for colname in data.columns:
        ci = _get_columninfo_from_columns_default(colname, columns_default)
        if ci:
            columns.append(ci)
        else:
            columns.append({"name": colname, "type": type_default})
    return columns


def _handle_progress(prog: int):
    print("\rprogress: {}%".format(prog), end='')
    return False


if __name__ == "__main__":
    logger.init("getimpt")

    import warnings
    warnings.filterwarnings("ignore")

    topdir = dirname(dirname(dirname(__file__)))
    sys.path.insert(0, topdir)
    from featuretools.mkfeat.feat_importance import FeatureImportance
    from featuretools.mkfeat.error import Error

    if len(sys.argv) < 2:
        usage()
        exit(1)

    if not os.path.isfile(sys.argv[1]):
        logger.error("configuration file not found: {}".format(sys.argv[1]))
        exit(1)

    conf = load_conf(sys.argv[1])

    path_data = conf['data']['uri']
    path_label = None
    if 'columns' in conf['data']:
        columns_data = conf['data']['columns']
    else:
        columns_data = _build_columns_from_csv(path_data, conf['data']['columns_default'])
    columns_label = None

    if 'label' in conf:
        conf_label = conf['label']
        path_label = conf_label['uri']
        columns_label = conf_label['columns']
    impt = FeatureImportance(path_data, columns_data, path_label, columns_label, _handle_progress)
    err = impt.analyze()
    if err != Error.OK:
        logger.error("error: {}".format(err))
        exit(2)

    print()
    print(impt.get_importance())

    exit(0)
