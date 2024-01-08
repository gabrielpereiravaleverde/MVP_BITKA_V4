from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class BaseModel(ABC, BaseEstimator, RegressorMixin):
    def __init__(self, random_state = 42):
        self.random_state = random_state

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs):
        pass   

    @abstractmethod
    def explain_global(self, **kwargs):
        pass

    @abstractmethod
    def explain_local(self, **kwargs):
        pass

    @abstractmethod
    def get_features(self):
        pass