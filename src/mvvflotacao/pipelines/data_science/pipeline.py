from kedro.pipeline import Pipeline, node, pipeline
from .nodes import calculate_metrics, create_valid_frame, data_test_predict, conformal_inference
from ...shared_code.utils import train_ebm_model, conformal_model, train_model
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
        for split_type in ['randomly', 'by_date']:
            nodes.extend([
                # Train the Model
                node(
                    func=lambda train_data,
                    train_data_s0,
                    params,
                    hyperparameters_params,
                    model_name=model_name,
                    train_mode=type: train_model(train_data=train_data,
                                                 train_data_s0=train_data_s0,
                                                 train_mode=train_mode,
                                                 model_name=model_name,
                                                 params=params,
                                                 hyperparameters_params=hyperparameters_params),
                    inputs=[f"{model_name}_{split_type}_train_data",
                            f"{model_name}_{split_type}_train_data_s0",
                            f"params:model_{model_name}",
                            f"params:train_ebm_model"],
                    outputs=f"ebm_{model_name}_{split_type}",
                    name=f"training_{model_name}_{split_type}_model"
                ),

                # Conformal Model
                node(
                    func=lambda model,
                    train_data,
                    train_data_s0,
                    params,
                    run_conformal_nodes,
                    model_name=model_name,
                    train_mode=type: conformal_model(model,
                                                     train_data=train_data,
                                                     train_data_s0=train_data_s0,
                                                     train_mode=train_mode,
                                                     params=params,
                                                     run_conformal_nodes=run_conformal_nodes,
                                                     name=model_name),
                    inputs=[f"ebm_{model_name}_{split_type}",
                            f"{model_name}_{split_type}_train_data",
                            f"{model_name}_{split_type}_train_data_s0",
                            f"params:model_{model_name}",
                            "params:run_conformal_nodes"],
                    outputs=f"conformal_model_{model_name}_{split_type}",
                    name=f"conformal_model_{model_name}_{split_type}"
                ),

                # Predict the Model
                node(
                    func=data_test_predict,
                    inputs=[f'{model_name}_{split_type}_test_data',
                            f'ebm_{model_name}_{split_type}'],
                    outputs=f"{model_name}_{split_type}_test_predicted",
                    name=f'predicting_{model_name}_{split_type}_test'
                ),

                # Calculate Metrics
                node(
                    func=calculate_metrics,
                    inputs=[f"{model_name}_{split_type}_test_predicted"],
                    outputs=f'{model_name}_{split_type}_ebm_score',
                    name=f"evaluate_model_node_{model_name}_{split_type}"
                ),

                # Conformal Inference
                node(
                    func=conformal_inference,
                    inputs=[f"{model_name}_{split_type}_test_predicted",
                            f"conformal_model_{model_name}_{split_type}",
                            "params:run_conformal_nodes"],
                    outputs=f"{model_name}_{split_type}_conformal_inference",
                    name=f'conformal_inference_{model_name}_{split_type}'
                ),

                # Create valid frame
                node(
                    func=create_valid_frame,
                    inputs=[f"{model_name}_{split_type}_train_data_s0",
                            f'{model_name}_{split_type}_conformal_inference',
                            "params:run_conformal_nodes"],
                    outputs=f"{model_name}_{split_type}_results",
                    name=f'validating_frame_{model_name}_{split_type}'
                ),
            ])

    return pipeline(nodes)
