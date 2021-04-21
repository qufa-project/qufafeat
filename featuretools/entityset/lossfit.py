from sklearn import linear_model
import numpy as np


def lossfit(entity):
    set_feature, set_target = get_features_target(entity)

    regs = {}
    for col in entity.df.columns:
        regs[col] = linear_model.LinearRegression()
        regs[col].fit(set_feature[col], set_target[col])

    for index, row in entity.df.iterrows():
        if not row.hasnans:
            continue
        col_lossy, features = get_lossy_features(row)
        if col_lossy is not None:
            est = regs[col_lossy].predict([features])
            entity.df[col_lossy][index] = est


def get_lossy_features(row):
    features = []
    col_lossy = None
    for col, val in row.iteritems():
        if np.isnan(val):
            if col_lossy is None:
                col_lossy = col
            else:
                return None, None
        else:
            features.append(val)

    return col_lossy, features


def get_features_target(entity):
    n_cols = len(entity.df.columns)
    set_feature = {}
    set_target = {}
    for col in entity.df.columns:
        set_feature[col] = []
        set_target[col] = []

    for index, row in entity.df.iterrows():
        if row.hasnans:
            continue

        set_feature[row.index[0]].append(row[1:])
        set_target[row.index[0]].append(row[0])

        for i in range(1, n_cols - 1):
            set_feature[row.index[i]].append(row[:i] + row[i + 1:])
            set_target[row.index[i]].append(row[i])

        set_feature[row.index[n_cols - 1]].append(row[:n_cols - 1])
        set_target[row.index[n_cols - 1]].append(row[n_cols - 1])

    return set_feature, set_target
