from abc import ABC, abstractmethod
import pandas as pd 
import numpy as np 
import pickle
from itertools import product
import optuna

class ObjectiveFunction(ABC):
    def __init__(self, fixed_data):
        self.fixed_data = fixed_data

    def build_full_solution(self, s):
        fixed = self.fixed_data.reset_index(drop = True)
        sol = s.reset_index(drop = True)
        if len(s) > len(self.fixed_data):
            fixed = fixed.loc[fixed.index.repeat(len(sol))].reset_index(drop=True)
        elif len(s) < len(self.fixed_data):
            sol = sol.loc[sol.index.repeat(len(fixed))].reset_index(drop=True)
        fixed[s.columns] = sol
        return fixed

    @abstractmethod
    def evaluate_solution(self, s) -> float:
        pass

class ModelEvaluate(ObjectiveFunction):
    def __init__(self, model,  fixed_data):
        super().__init__(fixed_data)
        self.model = model

    def evaluate_solution(self, s) -> float:
        y_hat = -1 * self.model.predict(s) # -1 in order to minimize this parameter
        return y_hat
    
class ModelPenaltyEvaluate(ModelEvaluate):
    def __init__(self, params, fixed_data, model_of, model_penalty):
        super().__init__(model_of, fixed_data)
        self.model_penalty = model_penalty
        self.params = params

    def evaluate_solution(self, s) -> float:
        y_hat = super().evaluate_solution(s)
        of = y_hat + self.params['penalty'] * self.model_penalty.predict(s)
        return of
    
class FeatureEngineering:
    """
    param example: 
        decision_variable_feature_engineering:
            processing:
            VAZAO_AR_ROUGHER_COND_sqrt: 
                feature: VAZAO_AR_ROUGHER_COND
                function: sqrt
            Espumante (g/t)_CD_log: 
                feature: Espumante (g/t)_CD
                function: log
            ESPUMA_ROUGHER_COND_pow2: 
                feature: ESPUMA_ROUGHER_COND
                function: pow2
            interactions: 
            interaction1:
                - VAZAO_AR_ROUGHER_COND
                - Espumante (g/t)_CD
    """
    def __init__(self, params):
        self.params = params
        self.functions = {'sqrt' : np.sqrt,
                          'pow2' : lambda x: np.power(x, 2), 
                          'log' : np.log}

    def build_features(self, space):
        if self.params is None:
            return space
        
        if "processing" in self.params.keys():
            for k, par in self.params['processing'].items():
                space[k] = self.functions[par['function']](space[par['feature']])

        if "interactions" in self.params.keys():
            for k, feats in self.params['interactions'].items():
                space[k] = [1] * len(space[feats[0]])
                for f in feats:
                    space[k] *= space[f]
        return space

class OptimizerInterface(ABC): 
    def __init__(self, decision_variables, objective_function: ObjectiveFunction, feature_engineering = None):
        self.decision_variables = decision_variables
        self.of = objective_function
        if feature_engineering is None:
            self.fe = FeatureEngineering(params=None)
        else: 
            self.fe = feature_engineering

    def get_optimal_solution(self, s):
        if "fitness" not in s.columns:
            raise Exception("Solution is not evaluated")
        min_value = s['fitness'].min()
        return s[s['fitness'] == min_value].head(1)
    
    @abstractmethod
    def optimize(self):
        pass

class GridSearchOptimizer(OptimizerInterface):
    def __init__(self, decision_variables, objective_function: ObjectiveFunction, feature_engineering = None, max_combination_size = 10e6):
        super().__init__(decision_variables, objective_function, feature_engineering)
        self.max_combination_size = max_combination_size

    def build_feature(self, parameters):
        space = []
        if parameters['method'] == "linear":
            space = np.linspace(parameters['min'], parameters['max'], parameters['steps'])
        elif parameters['method'] == 'loglinear':
            space = np.exp(np.linspace(np.log(parameters['min'] + 0.1 if parameters['min'] == 0 else parameters['min']), np.log(parameters['max']), parameters['steps']))
        elif parameters['method'] == 'categorical':
            space = parameters['categories']
        else:
            raise Exception(f"{parameters['method']} not implemented for GridSearchOptimizer")
        return space

    def combine_arrays(self, *arrays):
        return [list(combination) for combination in product(*arrays)]
    
    def build_grid(self):
        arrays = []
        columns = []
        comb_size = 1
        for f, parameters in self.decision_variables.items():
            arrays.append(self.build_feature(parameters))
            columns.append(f)
            comb_size *= len(arrays[-1])

        if comb_size >= self.max_combination_size:
            raise Exception('Search space exceeded max size')

        combination = self.combine_arrays(*arrays)
        return self.fe.build_features(pd.DataFrame(combination, columns = columns, index = range(len(combination))))
    
    def optimize(self):
        data = self.build_grid()
        fitness = self.of.evaluate_solution(self.of.build_full_solution(data))
        data['fitness'] = fitness
        return self.get_optimal_solution(data)
    

class ItGridSearchOptimizer(GridSearchOptimizer):
    def __init__(self, decision_variables, objective_function: ObjectiveFunction, feature_engineering = None, max_combination_size = 10e6):
        super().__init__(decision_variables, objective_function, feature_engineering, max_combination_size)

    def combine_arrays(self, *arrays):
        result = []
        for i, comb in enumerate(product(*arrays)):
            result.append(list(comb))
            if (i + 1) % self.max_combination_size == 0:
                yield result
                result = []

        if len(result) > 0:
            yield result

    def build_grid(self):
        arrays = []
        columns = []
        comb_size = 1
        for f, parameters in self.decision_variables.items():
            arrays.append(self.build_feature(parameters))
            columns.append(f)
            comb_size *= len(arrays[-1])

        for combination in self.combine_arrays(*arrays):
            yield self.fe.build_features(pd.DataFrame(combination, columns = columns, index = range(len(combination))))
    
    def optimize(self):
        opt_sol = pd.DataFrame({'fitness' : [np.inf]}, index = [0])
        for data in self.build_grid():
            fitness = self.of.evaluate_solution(self.of.build_full_solution(data))
            data['fitness'] = fitness
            sol = self.get_optimal_solution(data)
            if sol['fitness'].values[0] < opt_sol['fitness'].values[0]:
                opt_sol = sol
        return opt_sol

class BayesianOptimizer(OptimizerInterface):
    def __init__(self, decision_variables, objective_function: ObjectiveFunction, feature_engineering = None, max_time_min: int=1, max_it: int=100):
        super().__init__(decision_variables, objective_function, feature_engineering)
        self.max_time_min = max_time_min
        self.max_it = max_it
        self.study = None
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        self.progress_callback = None

    def __build_feature(self, trial, feature, parameters):
        space = {}
        if parameters['method'] == "linear":
            space = trial.suggest_float(feature, parameters['min'], parameters['max']),
        elif parameters['method'] == 'loguniform':
            space = trial.suggest_loguniform(feature, parameters['min'], parameters['max'])
        elif parameters['method'] == 'uniform':
            space = trial.suggest_uniform(feature, parameters['min'], parameters['max'])
        elif parameters['method'] == 'categorical':
            space = trial.suggeest_categorical(feature, parameters['categories'])
        else:
            raise Exception(f"{parameters['method']} not implemented for BayesianOptimizer")
        return space
    
    def set_progress_callback(self, progress_callback):
        self.progress_callback = progress_callback

    def optimize(self):
        def objective(trial):
            space = {}
            for f, parameters in self.decision_variables.items():
                space[f] = self.__build_feature(trial, f, parameters)
                
            if self.progress_callback:
                self.progress_callback(trial.number / self.max_it)
            result = self.of.evaluate_solution(self.of.build_full_solution(self.fe.build_features(pd.DataFrame(space, index = [0]))))
            if isinstance(result, np.ndarray):
                result = np.mean(result)

            return result
        
        self.study = optuna.create_study(direction = 'minimize')
        self.study.optimize(objective, timeout = self.max_time_min * 60, n_trials = self.max_it, n_jobs = -1)
        trial = self.study.best_trial
        results = trial.params
        results['fitness'] = trial.value
        return pd.DataFrame(results, index = [0])

class GAOptimizer(OptimizerInterface):
    def __init__(self, decision_variables, objective_function: ObjectiveFunction):
        super().__init__(decision_variables, objective_function)
