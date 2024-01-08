from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *
import logging
logger = logging.getLogger(__name__)

def create_pipeline(**kwargs) -> Pipeline:
    catalog = kwargs['catalog']
    params = catalog.datasets.parameters.load()
    if "models_to_solve" in params.keys():
        all_models = params['models_to_solve']
    else:
        logger.warning("models_to_solve not found in parameters. Falling back to default model conc_cd")
        all_models = ['conc_cd']

    nodes = []

    for model_name in all_models:
        nodes.extend(
            [
                node(
                    func=simulate,
                    inputs=[f'{model_name}_randomly_train_data', f"ebm_{model_name}_randomly", f"params:{model_name}_opt_sim_params"],
                    outputs=f"{model_name}_simulations",
                    name=f"simulate_{model_name}_model"
                ),

                node(
                    func=process_results,
                    inputs=[f"{model_name}_simulations", f"params:{model_name}_opt_sim_params"],
                    outputs=f"{model_name}_simulations_plot",
                    name=f"process_{model_name}_results"
                ),
            ]
        )
    return Pipeline(nodes)
