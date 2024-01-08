import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from . Explaining import Explainer
from .Interface import BaseModel
import itertools

class EnsembleEstimator(BaseModel):
    def __init__(self, estimators, features = None, weights = None):
        self.estimators = estimators

        if features is None:
            self.features = [None] * len(estimators)
        else:
            self.features = features

        if weights is None:
            self.weights = np.ones(len(estimators))
        else:
            self.weights = weights

        self.all_features = list(set(itertools.chain.from_iterable(features)))

    def fit(self, X,y):
        for estimator, feature, weight in zip(self.estimators, self.features, self.weights):
            if feature is None:
                estimator.fit(X,y)
            else:
                estimator.fit(X[feature], y)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for estimator, feature, weight in zip(self.estimators, self.features, self.weights):
            if feature is None:
                pred = estimator.predict(X)
            else:
                pred = estimator.predict(X[feature])
            predictions += weight * pred
        predictions /= sum(self.weights)

        return predictions
    
    def explain_local(self, X):
        exp = Explainer()
        return exp.explain_local(self.estimators, X)

    def explain_global(self):
        exp = Explainer()
        return exp.explain_global(self.estimators)

    def get_features(self):
        return self.all_features
    
