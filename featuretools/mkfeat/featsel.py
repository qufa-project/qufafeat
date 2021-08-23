from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from pandas import DataFrame

from featuretools import selection

from elapsed_time import ElapsedTime

THRESHOLD = 0.02


def select_features(df: DataFrame, features, n_feats, elapsed_time: ElapsedTime):
    df_new = DataFrame()
    features_new = []
    f_names_check = []

    for feat in features:
        name = feat.get_name()
        if feat.variable_type.type_string == "numeric" or feat.variable_type.type_string == "boolean":
            sel = VarianceThreshold()
            scaler = MinMaxScaler()
            try:
                arr = df[name].values.reshape(len(df), 1)
                scaler.fit(arr)
                sel.fit(scaler.transform(arr))
                if sel.variances_[0] >= THRESHOLD:
                    f_names_check.append(name)
                    df_new[name] = df[name]
                    features_new.append(feat)
            except ValueError:
                pass
        else:
            df_new[name] = df[name]
            features_new.append(feat)

    elapsed_time.mark()
    df_new, features_new = selection.remove_highly_correlated_features(df_new, features_new,
                                                                       features_to_check=f_names_check)

    df_new, features_new = _select_highvar_features(df_new, features_new, n_feats)
    return df_new, features_new


def _select_highvar_features(df: DataFrame, features, n_feats):
    name_vars = []
    for feat in features:
        name = feat.get_name()
        if feat.variable_type.type_string == "numeric" or feat.variable_type.type_string == "boolean":
            sel = VarianceThreshold()
            scaler = MinMaxScaler()
            try:
                arr = df[name].values.reshape(len(df), 1)
                scaler.fit(arr)
                sel.fit(scaler.transform(arr))
                var = sel.variances_[0]
            except ValueError:
                var = 0
        else:
            var = 1
        name_vars.append((name, var))

    df_new = DataFrame()
    features_new = []
    n = 0
    for name_var in reversed(sorted(name_vars, key=lambda x: x[1])):
        df_new[name_var[0]] = df[name_var[0]]
        features_new.append(features[df.columns.get_loc(name_var[0])])
        n += 1
        if n >= n_feats or name_var[1] < THRESHOLD:
            break
    return df_new, features_new
