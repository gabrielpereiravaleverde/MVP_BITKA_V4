from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_rawdata, preprocess_reagentes, preprocess_laboratorio
from .nodes import preprocess_blend, pivoting_blend, merge_raw_data, filter_data, agg_by_3h, add_new_features
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    catalog = kwargs['catalog']
    params = catalog.datasets.parameters.load()
    
    return pipeline(
        [
            node(
                func=preprocess_rawdata,
                inputs=["balanco_de_massas_raw",
                        "params:raw_data_version"],
                outputs="balanco_de_massas_pre",
                name="pre_balanco_de_massas_node",
                tags=['web_app']
            ),
            node(
                func=preprocess_laboratorio,
                inputs=["laboratorio_raw",
                        "params:raw_data_version"],
                outputs="laboratorio_pre",
                name="pre_laboratorio_node",
                tags=['web_app']
            ),
            node(
                func=preprocess_laboratorio,
                inputs=["laboratorio_raiox_raw",
                        "params:raw_data_version"],
                outputs="laboratorio_raiox_pre",
                name="pre_laboratorio_raiox_node",
                tags=["web_app"]
            ),
            node(
                func=preprocess_rawdata,
                inputs=["carta_controle_pims_raw",
                        "params:raw_data_version"],
                outputs="carta_controle_pims_pre",
                name="pre_carta_controle_pims_node",
                tags=['web_app']
            ),
            node(
                func=preprocess_reagentes,
                inputs=["reagentes_raw",
                        "params:raw_data_version"],
                outputs="reagentes_pre",
                name="pre_reagentes_node",
                tags=['web_app']
            ),
            node(
                func=preprocess_blend,
                inputs=["blend_raw",
                        "params:raw_data_version"],
                outputs="blend_pre",
                name="pre_blend_node",
                tags=['web_app']
            ),
            node(
                func=pivoting_blend,
                inputs=["blend_pre"],
                outputs="blend_pivot",
                name="pivoting_blend_node",
                tags=['web_app']
            ),
            node(
                func=merge_raw_data,
                inputs={
                    'metadata': 'metadata',
                    'balanco_de_massas': 'balanco_de_massas_pre',
                    'laboratorio': 'laboratorio_pre',
                    'laboratorio_raiox': 'laboratorio_raiox_pre',
                    'carta_controle_pims': 'carta_controle_pims_pre',
                    'reagentes': 'reagentes_pre',
                    'blend': 'blend_pivot'
                },
                outputs='merged_raw_data',
                name='merge_raw_data_node',
                tags=['web_app']
            ),
            node(
                func=filter_data,
                inputs=['merged_raw_data'],
                outputs='filtered_data',
                name='filter_data_node',
                tags=['test_models']
            ),
            node(
                func=agg_by_3h,
                inputs=['filtered_data',
                        'params:measures'],
                outputs='aggregated_data',
                name='agg_by_3h_node',
                tags=['test_models']
            ),

            node(
                func=add_new_features,
                inputs=['aggregated_data'],
                outputs='aggregated_data_with_new_features',
                name='add_new_features_node'
            ),
            
            node(
                func=split_data,
                inputs=[f'aggregated_data_with_new_features',
                        'random_indexes',
                        "params:split_data"],
                outputs=['train_data_randomly',
                         'test_data_randomly',
                         'train_data_by_date',
                         'test_data_by_date'],
                name=f'split_data_node'
            )
        ]
    )
