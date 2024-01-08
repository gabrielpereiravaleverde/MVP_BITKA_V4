"""mvvflotacao
"""

__version__ = "0.1"

from interpret.glassbox import ExplainableBoostingRegressor
from catboost import CatBoostRegressor

ExplainableBoostingRegressor.get_features = lambda self: self.feature_names_in_
CatBoostRegressor.get_features = lambda self: self.feature_names_
