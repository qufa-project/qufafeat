from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from pandas import DataFrame

from featuretools import selection

THRESHOLD = 0.02


def select_features(df: DataFrame, features):
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

    df_new, features_new = selection.remove_highly_correlated_features(df_new, features_new,
                                                                       features_to_check=f_names_check)
    return df_new, features_new
