import pandas as pd
from typing import List
from interpret.glassbox import ExplainableBoostingRegressor

class Explainer: 
    """
        Generic class to get local/global explanations to every method
        Need to implement to other types of models
    """
    def __init__(self):
        self.__local_exp = {ExplainableBoostingRegressor : Explainer.__ebm_local}
        self.__global_exp = {ExplainableBoostingRegressor : Explainer.__ebm_global}

    def explain_local(self, estimators, X):
        all_scores = []
        for estimator in estimators: 
            if type(estimator) in self.__local_exp.keys():
                scores = self.__local_exp[type(estimator)](estimator, X)
                all_scores.append(scores)
            else:
                method = getattr(estimator, "explain_local", None)
                if callable(method):
                    scores = method(X)
                    all_scores.append(scores)    
        
        score = pd.concat(all_scores, ignore_index = True)
        result = score.groupby("index")[[x for x in score.columns if x != "index"]].mean().reset_index()
        return result

    def explain_global(self, estimators):
        all_scores = []
        for estimator in estimators: 
            if type(estimator) in self.__global_exp.keys():
                scores = self.__global_exp[type(estimator)](estimator)
                all_scores.append(scores)
            else:
                method = getattr(estimator, "explain_global", None)
                if callable(method):
                    scores = method()
                    all_scores.append(scores)    
            
        score = pd.concat(all_scores, ignore_index = True)
        cols = [x for x in score.columns if x != "index"]
        return pd.DataFrame(dict(zip(cols, score[cols].mean())), index = [0])
    
    @staticmethod
    def __ebm_local(estimator, X):
        local = estimator.explain_local(X)
        scores = [x['scores'] for x in local._internal_obj['specific']]
        scores = pd.DataFrame(scores, columns = local._internal_obj['specific'][0]['names'], index = range(len(local._internal_obj['specific'])))
        scores['index'] = X.index
        return scores
    
    @staticmethod
    def __ebm_global(estimator):
        global_exp = estimator.explain_global().data()
        scores = pd.DataFrame(dict(zip(global_exp['names'], [[x] for x in global_exp['scores']])), index = [0])
        return scores
    
class MultEBMExplainer:
    def __init__(self, estimators: List[ExplainableBoostingRegressor]):
        self.estimators = estimators
    
    def explain_local(self, X) -> pd.DataFrame:
        all_scores = []
        for estimator in self.estimators:
            local = estimator.explain_local(X)
            scores = [x['scores'] for x in local._internal_obj['specific']]
            scores = pd.DataFrame(scores, columns = local._internal_obj['specific'][0]['names'], index = range(len(local._internal_obj['specific'])))
            scores['index'] = X.index
            all_scores.append(scores)

        score = pd.concat(all_scores, ignore_index = True)
        return score.groupby("index")[[x for x in score.columns if x != "index"]].mean()