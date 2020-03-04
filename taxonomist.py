"""
Contains the machine learning code for Taxonomist

Authors:
    Emre Ates (1), Ozan Tuncer (1), Ata Turk (1), Vitus J. Leung (2),
    Jim Brandt (2), Manuel Egele (1), Ayse K. Coskun (1)
Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import check_is_fitted


def generate_features(timeseries, feature_tuples, trim=60):
    """ Generate features from timeseries

    Parameters
    ----------
    timeseries : pd.DataFrame[time, metric]
        DataFrame of metrics over time.

    feature_tuples : Iterable[(feature_name, feature_function)]
        List of feature name strings and feature functions.

    trim : int
        The amount of time to trim from both ends to remove startup and
        finalization steps.

    Returns
    -------
    features : Array[metric * feature_types]
        The calculated list of features.
    """
    if trim != 0:
        timeseries = timeseries[trim:-trim]
    features = []
    for col in timeseries.columns:
        for name, func in feature_tuples:
            features.append(pd.Series(
                name=name + '_' + col,
                data=func(timeseries[col])
            ))
    return pd.concat(features, axis=1)


class Taxonomist(OneVsRestClassifier):
    """ The main class implementing Taxonomist

    Parameters
    ----------
    estimator : estimator object
        The main estimator that the classifier is going to use.

    n_jobs : int
        Number of parallel processes to use.

    threshold : float
        The confidence threshold.
    """
    def __init__(self, estimator, n_jobs=1, threshold=0.9):
        self.threshold = threshold
        self.scaler = MinMaxScaler()
        super().__init__(estimator, n_jobs)

    def fit(self, X, y):
        norm_X = self.scaler.fit_transform(X)
        super().fit(norm_X, y)

    def decision_function(self, X):
        """ Reports the distance from the decision boundary
            for classifiers that support it.

        We need to override the `decision_function` from scikit-learn because
        we need a likelihood estimate for classifiers that only report
        `predict_proba`. After pull request
        [#10612](https://github.com/scikit-learn/scikit-learn/pull/10612)
        is merged, we can use the normal `decision_function` from sklearn.
        """
        check_is_fitted(self, 'estimators_')
        if len(X) == 0:
            return pd.DataFrame(data=[], index=X.index, columns=self.classes_)
        try:
            T = np.array([est.decision_function(X).ravel()
                          for est in self.estimators_]).T
        except AttributeError:
            T = np.array([e.predict_proba(X)[:, 1] * 2 - 1
                          for e in self.estimators_]).T
        if len(self.estimators_) == 1:
            T = T.ravel()
        return pd.DataFrame(data=T, columns=self.classes_, index=X.index)

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        norm_X = pd.DataFrame(data=self.scaler.transform(X), columns=X.columns,
                              index=X.index)
        probas = self.decision_function(norm_X)
        small_idx = probas.max(axis=1) < self.threshold
        maxes = pd.DataFrame(data=np.argmax(probas.values, axis=1).astype(str),
                             index=probas.index)
        for i, cls in enumerate(self.classes_):
            maxes[maxes == str(i)] = cls
        maxes.loc[small_idx] = 'unknown'
        return maxes
