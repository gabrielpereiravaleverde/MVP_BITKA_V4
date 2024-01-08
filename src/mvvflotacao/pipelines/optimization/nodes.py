"""
This is a boilerplate pipeline 'optimization'
generated using Kedro 0.18.14
"""
import matplotlib.pyplot as plt
from ...shared_code.optimization import simulation, optimizer, optimizer_utils
from ...shared_code.modeling import Ensemble


def simulate(data, model, params):
    decision_variables_gs = {}
    for f in params['decision_variables']: 
        decision_variables_gs[f] = {'method' : 'linear', 'min' : data[f].quantile(0.1), 'max' : data[f].quantile(0.9), 'steps' : params['gs_steps']}

    scenario = simulation.ClusterScenarios(num_scenarios = 30, features = params['decision_variables'] + ['target'] + params['scenario_features'])

    if "decision_variable_feature_engineering" in params.keys():
        fe = optimizer.FeatureEngineering(params['decision_variable_feature_engineering'])
    else:
        fe = optimizer.FeatureEngineering(None)

    simulator = simulation.Simulation(scenario = scenario, 
                        baseline_searcher_builder = lambda evaluator: optimizer.ItGridSearchOptimizer(decision_variables = decision_variables_gs, 
                                                                                                    objective_function = evaluator,
                                                                                                    feature_engineering = fe),
                        searcher_builder = lambda evaluator: optimizer.BayesianOptimizer(decision_variables = decision_variables_gs, 
                                                                                         objective_function = evaluator, 
                                                                                         max_time_min = params['bayesian_max_time'], 
                                                                                         max_it = params['bayesian_max_it'],
                                                                                         feature_engineering = fe),
                        evaluator_builder = lambda fixed_data: optimizer.ModelEvaluate(model = model, fixed_data = fixed_data),
                        decision_variables = params['decision_variables'],
                        target = 'target'
                        )

    results = simulator.simulate(data = data)
    return results

def process_results(results, params):
    report = optimizer_utils.SimulationResults(results = results, params = params['decision_variables'])
    report.full_plot()
    return plt

