from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from pandas import DataFrame


THRESHOLD = 0.2


def select_features(df: DataFrame, features):
    df_new = DataFrame()
    features_new = []

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
                    df_new[name] = df[name]
                    features_new.append(feat)
            except ValueError:
                pass
        else:
            df_new[name] = df[name]
            features_new.append(feat)

    return df_new, features_new
