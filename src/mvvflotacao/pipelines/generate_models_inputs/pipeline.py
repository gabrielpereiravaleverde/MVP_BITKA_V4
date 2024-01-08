import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_dataset_for_conc_cd, generate_dataset_for_rec_global
from .nodes import remove_autocorrelation
import logging
logger = logging.getLogger(__name__)

generate_dataset_for = {
    'conc_cd': generate_dataset_for_conc_cd,
    'rec_global': generate_dataset_for_rec_global
}


def remove_autocorrelation_s0(df: pd.DataFrame) -> pd.DataFrame:
    return remove_autocorrelation(df=df, start=0)


def remove_autocorrelation_s1(df: pd.DataFrame) -> pd.DataFrame:
    return remove_autocorrelation(df=df, start=1)


def remove_autocorrelation_s2(df: pd.DataFrame) -> pd.DataFrame:
    return remove_autocorrelation(df=df, start=2)


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
        if model_name not in generate_dataset_for.keys():
            logger.warning(f"There's no implementation to generate dataset for {model_name}")
            continue

        nodes.extend([

            node(
                func=generate_dataset_for[model_name],
                inputs=['train_data_randomly',
                        'test_data_randomly',
                        'train_data_by_date',
                        'test_data_by_date',
                        f'params:generate_dataset_for_{model_name}',
                        'metadata'],
                outputs=[f'{model_name}_randomly_train_data',
                         f'{model_name}_randomly_test_data',
                         f'{model_name}_by_date_train_data',
                         f'{model_name}_by_date_test_data',
                         f'{model_name}_full_data_corrected_type'],
                name=f'generate_dataset_for_{model_name}_random_node'
            ),

            node(
                func=remove_autocorrelation_s0,
                inputs=f'{model_name}_full_data_corrected_type',
                outputs=f'{model_name}_full_data_without_correlation_s0',
                name=f'remove_autocorrelation_full_data_{model_name}_s0_node'
            ),

        ])
        for split_method in ['randomly', 'by_date']:
            nodes.extend([

                node(
                    func=remove_autocorrelation_s0,
                    inputs=f'{model_name}_{split_method}_train_data',
                    outputs=f'{model_name}_{split_method}_train_data_s0',
                    name=f'remove_autocorrelation_{model_name}_{split_method}_s0_node'
                ),

                node(
                    func=remove_autocorrelation_s1,
                    inputs=f'{model_name}_{split_method}_train_data',
                    outputs=f'{model_name}_{split_method}_train_data_s1',
                    name=f'remove_autocorrelation_{model_name}_{split_method}_s1_node'
                ),

                node(
                    func=remove_autocorrelation_s2,
                    inputs=f'{model_name}_{split_method}_train_data',
                    outputs=f'{model_name}_{split_method}_train_data_s2',
                    name=f'remove_autocorrelation_{model_name}_{split_method}_s2_node'
                ),

            ])

    return Pipeline(nodes)
