from kedro.pipeline import Pipeline, node, pipeline
from .nodes import optimize_ebm_model
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

    return pipeline([
        node(
            func=lambda s0, s1, s2, params_opt, params_build_model, model_name=model_name: optimize_ebm_model(
                s0, s1, s2, model_name, params_opt=params_opt, params_build_model = params_build_model),
            inputs=[f'{model_name}_by_date_train_data_s0',
                    f'{model_name}_by_date_train_data_s1',
                    f'{model_name}_by_date_train_data_s2',
                    'params:optimize_ebm_model',
                    f'params:model_{model_name}'],
            outputs=[f'{model_name}_hyperparameters',
                     f'{model_name}_latest_hyperparameters'],
            name=f'optimize_ebm_model_for_{model_name}'
        )
        for model_name in all_models
    ])