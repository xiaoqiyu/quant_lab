# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : rpyxqi@gmail.com
# @file      : pipeline.py

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import _incremental_mean_and_var


class OutlierRemoval(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = None

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y: Passthrough for ``Pipeline`` compatibility.
        """
        self.mean_, self.var_, self.n_samples_seen_ = \
            _incremental_mean_and_var(X, self.mean_, self.var_,
                                      self.n_samples_seen_)
        return self

    def transform(self, X, y=None, copy=None):
        pass


num_pipeline = Pipeline([
    ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('label_binarizer', LabelBinarizer())
])

# feature_selection_pipeline = Pipeline([
#     {'feature_selection_ic', SelectFromModel(LinearSVC(penalty='l1'))},
#     {'classification', RandomForestClassifier()}
# ])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

if __name__ == '__main__':
    x = [[1, 2, 3], [4, 5, 6]]
    x1 = num_pipeline.fit_transform(x)
    print(x1)
