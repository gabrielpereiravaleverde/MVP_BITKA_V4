import pandas as pd
import numpy as np
from pathlib import Path
from interpret.glassbox import ExplainableBoostingRegressor
from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc, AbsErrorErrFunc, RegressorNormalizer
from abc import ABC, abstractmethod
from tqdm import tqdm

from mapie.metrics import regression_coverage_score
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.subsample import Subsample
from .Explaining import Explainer
from .Interface import BaseModel

class BaseConformalModel(BaseModel):
    def __init__(self, significance = 0.05, random_state = 42):
        self.significance = significance
        self.random_state = random_state

class BootstrappingModel(BaseConformalModel):
    def __init__(self, base_model, num_bootstrap, significance = 0.05):
        super().__init__(significance=significance)
        self.base_model = base_model
        self.num_bootstrap = num_bootstrap
        self.trained_models = []
        self.trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.features = X.columns
        self.trained_models = []
        for i in tqdm(range(self.num_bootstrap)):
            idx = X.sample(n = len(X), replace = True).index
            X_train = X.loc[idx]
            y_train = y.loc[idx]
            self.trained_models.append(self.base_model.fit(X_train, y_train).copy())
        self.trained = True
    
    def predict(self, X, **kwargs):
        if ~self.trained or len(self.trained_models) == 0: 
            raise Exception("Model is not trained")
        
        results = []
        for model in self.trained_models:
            pred = model.predict(X)
            results.append(pd.DataFrame({'idx' : X.index, 'Prediction' : pred}))
        
        results = pd.concat(results, ignore_index = True)
        pred = results.groupby("idx").agg({'Prediction' : ['mean', lambda x: x.quantile(self.significance / 2.0), lambda x: x.quantile(1 - self.significance / 2.0)]})
        pred.columns = ['Prediction', 'Min', 'Max']
        return pred[['Prediction', 'Min', 'Max']].values
    
    def get_features(self):
        return self.features
    
    def explain_global(self):
        exp = Explainer()
        return exp.explain_global(self.trained_models)
    
    def explain_local(self, X):
        exp = Explainer()
        return exp.explain_local(self.trained_models, X)

class InductiveConformal(BaseConformalModel):
    def __init__(self, base_model, significance):
        super().__init__(significance = significance)
        self.base_model = base_model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise Exception("not implemented")
        # underlying_model = RegressorAdapter(self.base_model)
        # nc = RegressorNc(underlying_model, AbsErrorErrFunc())
        # self.icp = IcpRegressor(nc)
        # self.icp.fit(X, y)
        # self.icp.calibrate(X_cal, y_cal)

    def predict(self, X, **kwargs):
        y_pred = self.icp.predict(X.values, significance = self.significance)
        result = pd.DataFrame({'Max' : y_pred[:,1], 'Min' : y_pred[:,0], 'Prediction' : np.mean(y_pred, axis = 1)}, index = X.index)
        return result[['Prediction', 'Min', 'Max']].values
    
class MapieConformal(BaseConformalModel):
    def __init__(self, base_model, significance, params):

        """
        params examples: 
            naive ->                                dict(method="naive"),
            jackknife ->                            dict(method="base", cv=5),
            jackknife_plus ->                       dict(method="plus", cv=5),
            jackknife_minmax ->                     dict(method="minmax", cv=5),
            cv ->                                   dict(method="base", cv=10),
            cv_plus ->                              dict(method="plus", cv=10),
            cv_minmax ->                            dict(method="minmax", cv=10),
            jackknife_plus_ab ->                    dict(method="plus", cv=Subsample(n_resamplings=50)),
            jackknife_minmax_ab ->                  dict(method="minmax", cv=Subsample(n_resamplings=50)),
            conformalized_quantile_regression ->    dict(method="quantile", cv="split", alpha=0.05)
        """
        super().__init__(significance = significance)
        self.params = params
        self.base_model = base_model
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fitted = True
        self.features = X.columns
        if self.params['method'] == "quantile":
            self.model = MapieQuantileRegressor(self.base_model, **self.params)
            self.model.fit(X, y, random_state = self.random_state)
        else:
            self.model = MapieRegressor(self.base_model, **self.params)
            self.model.fit(X, y)
            
    def predict(self, X):
        if not self.fitted: 
            return None
        if self.params['method'] == "quantile":
            y_pred, y_interval = self.model.predict(X)
        else:
            y_pred, y_interval = self.model.predict(X, alpha = self.significance)

        result = pd.DataFrame({'Max' : y_interval[:,1,0], 'Min' : y_interval[:,0,0], 'Prediction' : y_pred}, index = X.index)
        return result[['Prediction', 'Min', 'Max']].values
    
    def explain_local(self, X):
        if not self.fitted: 
            return None
        exp = Explainer()
        return exp.explain_local(self.model.estimator_.estimators_, X)

    def explain_global(self):
        if not self.fitted: 
            return None
        exp = Explainer()
        return exp.explain_global(self.model.estimator_.estimators_)
    
    def get_features(self):
        if not self.fitted:
            return []
        return self.features
        
