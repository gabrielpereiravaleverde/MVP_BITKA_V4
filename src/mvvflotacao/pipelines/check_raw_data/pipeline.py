from kedro.pipeline import Pipeline, node, pipeline
from typing import Any, Callable, Dict
from .nodes import check_rawdata

def _check_balanco_de_massas_raw(dfs: Dict[str, Callable[[], Any]], raw_data_version: str, previous_raw_data_version: str) -> None:
    check_rawdata(ds_name='balanco_de_massas', dfs=dfs, raw_data_version=raw_data_version, previous_raw_data_version=previous_raw_data_version)

def _check_reagentes_raw(dfs: Dict[str, Callable[[], Any]], raw_data_version: str, previous_raw_data_version: str) -> None:
    check_rawdata(ds_name='reagentes', dfs=dfs, raw_data_version=raw_data_version, previous_raw_data_version=previous_raw_data_version)

def _check_laboratorio_raw(dfs: Dict[str, Callable[[], Any]], raw_data_version: str, previous_raw_data_version: str) -> None:
    check_rawdata(ds_name='laboratorio', dfs=dfs, raw_data_version=raw_data_version, previous_raw_data_version=previous_raw_data_version)

def _check_laboratorio_raiox_raw(dfs: Dict[str, Callable[[], Any]], raw_data_version: str, previous_raw_data_version: str) -> None:
    check_rawdata(ds_name='laboratorio_raiox', dfs=dfs, raw_data_version=raw_data_version, previous_raw_data_version=previous_raw_data_version)

def _check_carta_controle_pims_raw(dfs: Dict[str, Callable[[], Any]], raw_data_version: str, previous_raw_data_version: str) -> None:
    check_rawdata(ds_name='carta_controle_pims', dfs=dfs, raw_data_version=raw_data_version, previous_raw_data_version=previous_raw_data_version)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=_check_balanco_de_massas_raw,
                inputs=["balanco_de_massas_raw", "params:raw_data_version", "params:previous_raw_data_version"],
                outputs=None,
                name="check_balanco_de_massas_raw",
            ),
            node(
                func=_check_reagentes_raw,
                inputs=["reagentes_raw", "params:raw_data_version", "params:previous_raw_data_version"],
                outputs=None,
                name="check_reagentes_raw",
            ),
            node(
                func=_check_laboratorio_raw,
                inputs=["laboratorio_raw", "params:raw_data_version", "params:previous_raw_data_version"],
                outputs=None,
                name="check_laboratorio_raw",
            ),
            node(
                func=_check_laboratorio_raiox_raw,
                inputs=["laboratorio_raiox_raw", "params:raw_data_version", "params:previous_raw_data_version"],
                outputs=None,
                name="check_laboratorio_raiox_raw",
            ),
            node(
                func=_check_carta_controle_pims_raw,
                inputs=["carta_controle_pims_raw", "params:raw_data_version", "params:previous_raw_data_version"],
                outputs=None,
                name="check_carta_controle_pims_raw",
            )
        ]
    )