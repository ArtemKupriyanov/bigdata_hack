import pandas as pd
import numpy as np
from tqdm import tqdm
import gc


class SemenovEncoding(object):
    def __init__(self, C=10):
        self.C = C

    def fit(self, data, y, features='all'):

        self.y = y
        if features == 'all':
            self.features = sorted(
                [i for i in data.columns if data[i].dtype == 'O'])
        else:
            self.features = features

        self.global_mean = np.mean(y)
        self.values = dict()
        data["target"] = y
        for feature in tqdm(self.features, desc="fitting"):
            groupby_feature = data.groupby([feature])
            current_mean = groupby_feature.target.mean()
            current_size = groupby_feature.size()
            if self.C != 'adaptive':
                C = self.C
            else:
                C = np.percentile(current_size.values, 5)
            feat_df = ((current_mean * current_size + self.global_mean * C) /\
                       (current_size + C)).fillna(self.global_mean)
            self.values[feature] = pd.DataFrame(feat_df,
                                                columns=["stas_%s" % feature],
                                                dtype=np.float64)
        data.drop(["target"], axis=1, inplace=True)
        return self.values

    def transform(self, data):
        features = [i for i in self.values if i in data.columns]
        for feature in tqdm(features, desc="merging"):
            data = pd.merge(data, self.values[feature], how="left",
                            left_on=feature, right_index=True)
        return data.fillna(self.global_mean)

    def fit_transform(self, data, y, features='all'):
        self.fit(data, y, features)
        return self.transform(data)