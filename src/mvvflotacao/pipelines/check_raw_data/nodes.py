from typing import Any, Callable, Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _check_column_names(new_df: pd.DataFrame, old_df: pd.DataFrame, ds_name: str) -> None:
    logger.info(f'Comparando nomes de colunas de {ds_name}')
    column_diff = False

    for c in new_df.columns:
        if c not in old_df.columns:
            column_diff = True
            logger.warning(f'A coluna {c} é nova. Não existia no dataframe anterior')

    for c in old_df.columns:
        if c not in new_df.columns:
            column_diff = True
            logger.warning(f'A coluna {c} deixou de existir no novo dataframe')

    if not column_diff:
        logger.info('Os nomes das colunas casam perfeitamente!')


def _check_column_types(new_df: pd.DataFrame, old_df: pd.DataFrame, ds_name: str) -> None:
    logger.info(f'Comparando tipos das colunas de {ds_name}')
    column_diff = False

    for c in new_df.columns:
        if c in old_df.columns and new_df[c].dtype != old_df[c].dtype:
            column_diff = True
            logger.warning(f'O tipo da coluna {c} mudou de {old_df[c].dtype} para {new_df[c].dtype}')

    if not column_diff:
        logger.info('Os tipos das colunas casam perfeitamente!')


def _check_hist_values(new_df: pd.DataFrame, old_df: pd.DataFrame, ds_name: str) -> None:
    logger.info(f'Comparando variações no histórico de dados de {ds_name}')
    key = 'DATA'
    count_diffs = 0

    merged_df = new_df.merge(
        old_df,
        how='outer',
        on=key,
        suffixes=('_new', '_old')
    )
    merged_df = merged_df[merged_df[key] <= old_df[key].max()]

    for column in new_df.columns:
        if column in old_df.columns and column != key:
            diff_df = merged_df[merged_df[f'{column}_new'] != merged_df[f'{column}_old']]
            
            # remove registros em que ambas as colunas são nulas
            diff_df = diff_df[[key, f'{column}_new', f'{column}_old']].dropna(
                subset = [f'{column}_new', f'{column}_old'],
                how='all'
            )

            if len(diff_df) > 0:
                count_diffs += len(diff_df)
                logger.warning(f'Ocorreram {len(diff_df)} variações no histórico de dados da coluna {column} de {ds_name}')
                logger.debug(diff_df.head(3))

    if count_diffs == 0:
        logger.info(f'Nenhuma diferença encontrada no histórico de dados de {ds_name}')
    else:
        logger.warning(f'Total de {count_diffs} divergências no histórico de {ds_name}')


def _treats_reagentes(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c[0] if c[1].startswith('Unnamed') else f'{c[0]}_{c[1]}' for c in df.columns]
    df['DATA'] = pd.to_datetime(
        df.apply(
            lambda x: x['DATA'] if str(x['Horário']) == 'nan' else f"{x['DATA'].strftime('%Y-%m-%d')} {x['Horário']}",
            axis=1
        )
    )

    return df


def check_rawdata(dfs: Dict[str, Callable[[], Any]], raw_data_version: str, previous_raw_data_version: str, ds_name: str) -> None:
    """
    Convert all columns with object dtype to string and return the processed dataframe.

    Args:
        dfs (Dict[str, Callable[[], Any]]): dict from partitions to load functions.
        raw_data_version (str): The new raw data version (YYYY_MM_DD-HH_MM_SS)
        previous_raw_data_version (str): The old raw data version (YYYY_MM_DD-HH_MM_SS)
        ds_name (str): The dataset name to be checked

    Returns:
        None
    """
    logger.info(f'Comparando {raw_data_version} em relação a {previous_raw_data_version}')

    logger.info('Carregando arquivos...')
    new_df = dfs[raw_data_version]()
    old_df = dfs[previous_raw_data_version]()

    if ds_name == 'reagentes':
        new_df = _treats_reagentes(new_df)
        old_df = _treats_reagentes(old_df)

    checks = [_check_column_names, _check_column_types, _check_hist_values]

    for _check in checks:
        _check(new_df=new_df, old_df=old_df, ds_name=ds_name)