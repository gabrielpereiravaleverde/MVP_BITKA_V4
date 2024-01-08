from ...shared_code.utils import get_metadata_df_from_dict
from datetime import timedelta
from typing import Any, Callable, Dict, Optional

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def read_data(data, data_version):

    if isinstance(data, dict):
        if data_version in data:
            logger.info(f'Carregando versão {data_version}.')
            return data[data_version]()
        else:
            raise KeyError(f'Versão de dados {data_version} não encontrada.')

    return data


def preprocess_rawdata(partitions, raw_data_version: Optional[str] = None) -> pd.DataFrame:
    """
    Convert all columns with object dtype to string and return the processed dataframe.

    Args:
        df (Dict[str, Callable[[], Any]] | pd.DataFrame): Raw dataframe to be processed.
        raw_data_version (Optional[str]): The version to be load (YYYY_MM_DD-HH_MM_SS). Only used if df is a dict.

    Returns:
        pd.DataFrame: Processed dataframe with object columns converted to string.
    """
    df = read_data(partitions, raw_data_version)

    df2 = df.convert_dtypes()
    object_columns = df2.select_dtypes(include=['object']).columns
    df2.loc[:, object_columns] = df2.loc[:, object_columns].astype(str)

    return df2


def preprocess_reagentes(partitions: Dict[str, Callable[[], Any]], raw_data_version: str) -> pd.DataFrame:
    """
    Preprocesses data read from a spreadsheet, combining the date and time columns,
    filtering the data, and filling in missing values.

    Args:
        partitions (Dict[str, Callable[[], Any]]): dict from partitions to load functions.
        raw_data_version (str): The version to be loaded (YYYY_MM_DD-HH_MM_SS)

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """

    df = read_data(partitions, raw_data_version)

    # Rename columns
    df.columns = [c[0] if c[1].startswith(
        'Unnamed') else f'{c[0]}_{c[1]}' for c in df.columns]

    # Reanalyze column dtypes
    df = df.convert_dtypes()

    # Drop rows with NaN values in 'Horário' (Time) column
    df.dropna(subset=['Horário'], inplace=True)

    # Combine 'DATA' (Date) and 'Horário' (Time) columns and convert into datetime
    df['DATA'] = pd.to_datetime(df['DATA'].astype(
        str) + ' ' + df['Horário'].astype(str))
    df.drop(columns=['Horário'], inplace=True)

    # Round the datetime column to the nearest hour
    df['DATA'] = df['DATA'].dt.round('H')

    # Filter data and any other operation you wish
    df = preprocess_rawdata(df)

    # Create a date range with a frequency of one hour
    date_range = pd.date_range(start=df['DATA'].min(), end=df['DATA'].max(), freq='H')

    # Create a dataframe with this date range
    df_dates = pd.DataFrame({'DATA': date_range})

    # Merge the original dataframe with the new date dataframe
    df_merged = df_dates.merge(df, on='DATA', how='left')
    df_merged = df_merged[df_merged['DATA'] > '2023-06-01 00:00:00']

    # Fill NaN values with the last valid preceding value
    for col in df_merged.columns:
        if col != 'DATA':
            df_merged[col] = df_merged[col].ffill()

    return df_merged


def preprocess_laboratorio(partitions: Dict[str, Callable[[], Any]], raw_data_version: str) -> pd.DataFrame:
    """
    Preprocess laboratory data by filtering out unreceived samples and adjusting the data timestamps.

    Args:
        partitions (Dict[str, Callable[[], Any]]): dict from partitions to load functions.
        raw_data_version (str): The version to be loaded (YYYY_MM_DD-HH_MM_SS)

    Returns:
        pd.DataFrame: Preprocessed laboratory data with adjusted timestamps.
    """

    df = read_data(partitions, raw_data_version)

    df2 = df[(df['CODIGO_AMOSTRA'] != 'NÃO RECEBIDA') & (
        df['CODIGO_AMOSTRA'] != 'NÃORECEBIDA')].copy()
    if 'ALIM_FLOT_CU_SOL' in df2.columns and 'ALIM_FLOT_CU_TOT' in df2.columns:
        # Calculate the ratio
        df2['RAZAO_CU_SOL'] = df2['ALIM_FLOT_CU_SOL'] / df2['ALIM_FLOT_CU_TOT']

        # Fill with 0 where either column is 0 or NaN
        mask = (df2['ALIM_FLOT_CU_SOL'].isna() | (df2['ALIM_FLOT_CU_SOL'] == 0) |
                df2['ALIM_FLOT_CU_TOT'].isna() | (df2['ALIM_FLOT_CU_TOT'] == 0))
        df2.loc[mask, 'RAZAO_CU_SOL'] = 0
    laboratorio = preprocess_rawdata(df2)

    # remove amostras estranhas
    laboratorio = laboratorio[(laboratorio['DATA'] !=
                               pd.to_datetime('2023-07-03 22:00:00'))]
    laboratorio = laboratorio[(laboratorio['DATA'] !=
                               pd.to_datetime('2023-07-11 22:00:00'))]
    laboratorio = laboratorio[(laboratorio['DATA'] !=
                               pd.to_datetime('2023-07-27 04:00:00'))]

    laboratorio_1hora = laboratorio.copy()
    laboratorio_2hora = laboratorio.copy()

    laboratorio_1hora['DATA'] = [(x - timedelta(hours=2))
                                 for x in laboratorio_1hora['DATA']]
    laboratorio_2hora['DATA'] = [(x - timedelta(hours=1))
                                 for x in laboratorio_2hora['DATA']]
    
    def keep_less_nulls(group):
        return group.loc[group.isnull().sum(axis=1).idxmin()]
    
    combined_df = pd.concat([laboratorio_1hora, laboratorio_2hora, laboratorio])
    
    return combined_df.groupby('DATA', as_index=False).apply(keep_less_nulls).reset_index(drop=True)


def preprocess_blend(partitions: Dict[str, Callable[[], Any]], raw_data_version: str) -> pd.DataFrame:
    """
    Preprocess the blend data.

    Args:
        partitions (Dict[str, Callable[[], Any]]): A dictionary containing partitions.
        raw_data_version (str): The raw data version.

    Returns:
        pd.DataFrame: Preprocessed blend data.
    """

    # Verificar se a chave raw_data_version existe no dicionário partitions

    data = read_data(partitions, raw_data_version)

    blend = preprocess_rawdata(data)

    blend["Data_Hora_Inicio"] = pd.to_datetime(
        blend["DataInicio"].astype(str) + " " + blend["HoraInicio"])
    blend["Data_Hora_Fim"] = pd.to_datetime(
        blend["DataFim"].astype(str) + " " + blend["HoraFim"])
    blend["Data_Hora_Fim - Ultima"] = blend["Data_Hora_Fim"]
    blend = preprocess_blend_with_categorization_v2(blend)

    blend["Origem_Area"] = blend["Origem_Area_CM"].apply(
        lambda x: "Frente de Lavra" if "FASE" in x else
                  "Pilha Sulfetado" if x == "PILHA SULFETADO" else
                  "Platô Britagem" if x == "PLATO BRITAGEM" else
                  "Pátio de Rompidos" if x == "PÁTIO DE ROMPIDOS" else None).astype(str)

    blend = blend.rename(columns={
        "Destino_Area_CM": "Destino_Area",
        "Destino_Sub_Area_CM": "Destino_Sub_Area",
        "Balanca_Manager": "Massa",
        "Origem_Sub_Area_CM": "Origem_Sub_Area",
        "Material_CM": "Tipo_Material",
        "el3": "Teor de Cu"
    })

    blend = blend[
        (~blend["Tipo_Ciclo"].isin(["Interrompido", "Em aberto", "Edit_Delete", np.nan])) &
        (blend["Teor de Cu"].notna()) &
        (blend["Destino_Area"] == "BRITADOR") &
        (blend["Destino_Sub_Area"] == "BRITADOR")]

    blend["Início da Hora"] = blend["Data_Hora_Fim"].dt.floor('H')

    return blend[["Data_Hora_Inicio", "Data_Hora_Fim", "Origem_Area_CM", "Origem_Area", "Origem_Sub_Area",
                  "Destino_Area", "Destino_Sub_Area", "Tipo_Material", "Massa", "Teor de Cu",
                  "Tipo_Movimentacao", "Data_Hora_Fim - Ultima", "Início da Hora", "Category",
                  "Tipologia", "Grade", "Fase", "Stock_Type"]]


def preprocess_blend_with_categorization_v2(df):
    """
    Updated preprocess function with categorization for 'Origem_Area', 'Tipologia', 'Grade', and 'Fase'.

    Args:
        df (pd.DataFrame): The DataFrame containing blend data.

    Returns:
        pd.DataFrame: The DataFrame with additional categorizations.
    """

    df['Category'] = df['Origem_Sub_Area_CM'].apply(
        lambda x: 'LIB' if x.startswith('LIB') else 'Non-LIB')

    # Placeholder para 'Tipologia' e 'Fase' - substitua pelo critério real
    df['Tipologia'] = df.apply(lambda row: 1 if row['Category'] == 'LIB' else 0, axis=1)
    df['Fase'] = df.apply(lambda row: 0 if 'FASE 1' in row['Origem_Area_CM']
                          else 1 if 'FASE 2' in row['Origem_Area_CM'] else None, axis=1)

    # Lógica de categorização para 'Grade'
    df['Grade'] = df['Material_CM'].apply(lambda x: 'Low Grade' if 'Sulfetado_LG' in x else
                                          'Medium Grade' if 'Sulfetado_MG' in x else
                                          'High Grade' if 'Sulfetado_HG' in x else
                                          'Super High Grade' if 'Sulfetado_SHG' in x else
                                          'Unknown')

    # Placeholder para tipos de estoque em 'Non-LIB' - substitua pelo critério real
    df['Stock_Type'] = df.apply(
        lambda row: 'Stock_Type_Criteria' if row['Category'] == 'Non-LIB' else None, axis=1)

    return df


def pivoting_blend(df):
    """
    Pivot and preprocess blend data to calculate the percentage of 'Tipo_Material' by hour and categorize into
    Estoque, Fase1, and Fase2, combining 'TIPO 1' with their respective main material types. Then reindex and fill
    missing values.

    Args:
        df (pd.DataFrame): Input DataFrame containing blend data.

    Returns:
        pd.DataFrame: Preprocessed and pivoted DataFrame with hourly percentages of 'Tipo_Material' categorized.
    """

    # Handling 'TIPO 1' in 'Tipo_Material'
    tipo_1_suffix = ' TIPO 1'
    for material in ['Sulfetado_LG', 'Sulfetado_MG', 'Sulfetado_HG', 'Sulfetado_SHG']:
        tipo_1_col = f'{material}{tipo_1_suffix}'
        if tipo_1_col in df['Tipo_Material'].unique():
            # Combine 'TIPO 1' values with their respective main material types
            df.loc[df['Tipo_Material'] == tipo_1_col, 'Tipo_Material'] = material

    # Initialize an empty DataFrame for the final result
    final_df = pd.DataFrame()

    # Define the categories and their corresponding filters
    categories = {
        'Estoque': df['Category'] != 'LIB',
        'Fase1': df['Fase'] == 0,
        'Fase2': df['Fase'] == 1
    }

    # Process each category
    for category, filter_condition in categories.items():
        filtered_df = df[filter_condition]

        # Group, sum and pivot
        grouped = filtered_df.groupby(['Início da Hora', 'Tipo_Material'])[
            'Massa'].sum().reset_index()
        pivot_table = grouped.pivot(index='Início da Hora',
                                    columns='Tipo_Material', values='Massa').fillna(0)

        # Calculate total mass per hour and percentages
        pivot_table['Total_Massa'] = pivot_table.sum(axis=1)
        for col in pivot_table.columns:
            if col != 'Total_Massa':
                pivot_table[col] = pivot_table[col] / pivot_table['Total_Massa'] * 100

        # Drop 'Total_Massa' and reset index
        pivot_table = pivot_table.drop(columns=['Total_Massa']).reset_index()

        # Rename columns with category suffix
        pivot_table = pivot_table.rename(
            columns=lambda x: f"{x}_{category}" if x != 'Início da Hora' else x)

        # Merge with final DataFrame
        if final_df.empty:
            final_df = pivot_table
        else:
            final_df = pd.merge(final_df, pivot_table, on='Início da Hora', how='outer')

    # Reindexing and filling missing values
    new_index = pd.date_range(start=final_df['Início da Hora'].min(
    ), end=final_df['Início da Hora'].max(), freq='H')
    final_df = final_df.set_index('Início da Hora').reindex(new_index).reset_index()
    final_df.rename(columns={'index': 'DATA'}, inplace=True)
    final_df = final_df.fillna(0)

    # Define sulfetado materials
    sulfetados = ['Sulfetado_LG', 'Sulfetado_MG', 'Sulfetado_HG', 'Sulfetado_SHG']

    for sulfetado in sulfetados:
        colunas_para_somar = []
        for cat in ['Estoque', 'Fase1', 'Fase2']:
            coluna = f'{sulfetado}_{cat}'
            if coluna not in final_df.columns:
                final_df[coluna] = 0
            colunas_para_somar.append(coluna)

        final_df[f'{sulfetado}'] = final_df[colunas_para_somar].sum(axis=1)

    return final_df


def pivoting_blend_cobre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot and preprocess blend data to calculate the hourly sum of copper quantity by origin.

    Args:
        df (pd.DataFrame): Input DataFrame containing blend data with 'Massa' and 'Teor de Cu'.

    Returns:
        pd.DataFrame: DataFrame with hourly sum of copper quantity for 'Estoque', 'Fase1', and 'Fase2'.
    """

    # Calcular a quantidade de cobre em cada registro
    df['Quantidade_Cu'] = df['Massa'] * df['Teor de Cu']

    # Inicializar um DataFrame para o resultado final
    result_df = pd.DataFrame(index=pd.date_range(
        start=df['Início da Hora'].min(), end=df['Início da Hora'].max(), freq='H'))

    # Definir as origens e suas respectivas condições de filtro
    origens = {
        'Estoque': df['Category'] != 'LIB',
        'Fase1': df['Fase'] == 0,
        'Fase2': df['Fase'] == 1
    }

    # Processar cada origem
    for origem, filter_condition in origens.items():
        # Filtrar o DataFrame pela condição da origem
        origem_df = df[filter_condition]

        # Agrupar por 'Início da Hora' e somar a 'Quantidade_Cu'
        origem_grouped = origem_df.groupby('Início da Hora')['Quantidade_Cu'].sum()

        # Adicionar os dados ao DataFrame de resultado
        result_df[f'Cobre_{origem}'] = origem_grouped

    # Preencher valores NaN com zeros e renomear o índice
    result_df.fillna(0, inplace=True)
    result_df.rename_axis('DATA', inplace=True)
    result_df.reset_index(inplace=True)

    return result_df


def filter_columns_before_merge(df: pd.DataFrame, table_name: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Filter columns from a DataFrame based on a provided metadata table.

    Args:
        df (pd.DataFrame): The input DataFrame.
        table_name (str): Name of the table to be used for filtering.
        metadata (pd.DataFrame): Metadata containing valid columns for each table.

    Returns:
        pd.DataFrame: DataFrame containing only the columns listed in metadata for the given table.
    """
    columns = metadata[metadata.table ==
                       table_name]['column'].dropna().drop_duplicates().to_list()
    columns = ['DATA'] + columns if ['DATA'] not in columns else columns

    return df[columns]


def merge_raw_data(**dfs) -> pd.DataFrame:
    """
    Merge multiple DataFrames using 'DATA' as a key column. 

    Args:
        **dfs: A variable-length keyword argument containing DataFrames and metadata.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """

    metadata = get_metadata_df_from_dict(dfs['metadata'])

    if len(metadata[['table', 'column']].drop_duplicates()) != len(metadata[['column']].drop_duplicates()):
        logger.warning('Existem colunas com nomes repetidos em bases distintas!')

    filtered_data = [
        filter_columns_before_merge(df=v, table_name=k, metadata=metadata)
        for k, v in dfs.items()
        if k != 'metadata'
    ]

    merged_df = filtered_data[0]

    for df in filtered_data[1:]:
        merged_df = merged_df.merge(
            df,
            on='DATA',
            how='outer'
        )
    return merged_df


def remove_index_before_and_after(df: pd.DataFrame, indexes_to_remove: pd.Index, n: int = 12) -> pd.DataFrame:
    """
    Remove specified indexes and 'n' indexes before and 'n' indexes after each of them.

    Args:
        df (pd.DataFrame): The input DataFrame.
        indexes_to_remove (pd.Index): Indexes to be removed.
        n (int, optional): Number of rows to be removed before and after each specified index.
                           Defaults to 12.

    Returns:
        pd.DataFrame: DataFrame after removing specified indexes and their neighbors.
    """
    indexes_with_borders = set()

    for idx in indexes_to_remove:
        for i in range(-n, n+1):  # This will range from -n to n, inclusive.
            if idx + i in df.index:
                indexes_with_borders.add(idx + i)

    return df.drop(list(indexes_with_borders), axis=0)


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a DataFrame by removing specific inconsistent periods.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: filtered DataFrame.
    """
    # remove período inconsistente
    filtered_df = df[(df['DATA'] < '2023-07-03') | (df['DATA'] > '2023-07-12')].copy()

    filtered_df = filtered_df.sort_values(
        by='DATA', ascending=True).reset_index(drop=True)

    # Removendo períodos de drenagem e preenchimento da planta
    indexes_to_remove = filtered_df[(filtered_df['REC_FLOT_CU'] == 0) | (
        filtered_df['CONC_FLOT_CU_TOT'] == 0)].index.tolist()
    filtered_df = remove_index_before_and_after(
        df=filtered_df, indexes_to_remove=indexes_to_remove, n=9)

    filtered_df.set_index('DATA', inplace=True)

    return filtered_df


def agg_by_3h(df: pd.DataFrame, measures: list) -> pd.DataFrame:
    """
    Aggregate DataFrame in 3-hour intervals.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Aggregated data in 3-hour intervals.
    """
    # remove período inconsistente
    agg_df = df.resample('3H').mean()

    # Iterate through numeric columns and calculate statistics
    for col in df.select_dtypes(include='number').columns:
        for measure in measures:
            if measure == 'min':
                agg_df[col + '_' + measure] = df[col].resample('3H').min()
            elif measure == 'max':
                agg_df[col + '_' + measure] = df[col].resample('3H').max()
            elif measure == 'median':
                agg_df[col + '_' + measure] = df[col].resample('3H').median()
            elif measure == 'diff_min_max':
                agg_df[col + '_' + measure] = df[col].resample(
                    '3H').apply(lambda x: x.max() - x.min())

    # Reset index for the final result
    agg_df = agg_df.reset_index()

    return agg_df


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    #Cleaner 1
    result_df['ABERT_VAL_FC_CLEANER_1'] = np.mean(result_df[["ABERT_VAL_FC_007", "ABERT_VAL_FC_008", "ABERT_VAL_FC_009"]], axis=1)
    result_df['GHU_CLEANER_I_FC_CLEANER_1'] = np.mean(result_df[["GHU_CLEANER_I_FC_07", "GHU_CLEANER_I_FC_08",]], axis=1)
    result_df['VAZAO_AR_CLEA_1_FC_CLEANER_1'] = np.mean(result_df[["VAZAO_AR_CLEA_1_FC_07", "VAZAO_AR_CLEA_1_FC_08", "VAZAO_AR_CLEA_1_FC_09"]], axis=1)

    #Cleaner 2
    result_df['ABERT_VAL_FC_CLEANER_2'] = np.mean(result_df[["ABERT_VAL_FC_010", "ABERT_VAL_FC_011", "ABERT_VAL_FC_012", "ABERT_VAL_FC_013",]], axis=1)
    result_df['GHU_CLEANER_II_FC_CLEANER_2'] = np.mean(result_df[["GHU_CLEANER_II_FC_10", "GHU_CLEANER_II_FC_11", "GHU_CLEANER_II_FC_12", "GHU_CLEANER_II_FC_13",]], axis=1)
    result_df['VAZAO_AR_CLEA_2_FC_CLEANER_2'] = np.mean(result_df[["VAZAO_AR_CLEA_2_FC_10", "VAZAO_AR_CLEA_2_FC_11", "VAZAO_AR_CLEA_2_FC_12", "VAZAO_AR_CLEA_2_FC_13"]], axis=1)

    #Cleaner scavenger
    result_df['ABERT_VAL_FC_CLEANER_SCAV'] = np.mean(result_df[["ABERT_VAL_FC_014", "ABERT_VAL_FC_015", "ABERT_VAL_FC_016", "ABERT_VAL_FC_017", "ABERT_VAL_FC_018"]], axis=1)
    result_df['GHU_SCAV_CLEANER_FC_CLEANER_SCAV'] = np.mean(result_df[["GHU_SCAV_CLEANER_FC_14", "GHU_SCAV_CLEANER_FC_15", "GHU_SCAV_CLEANER_FC_16", "GHU_SCAV_CLEANER_FC_17", "GHU_SCAV_CLEANER_FC_18"]], axis=1)
    result_df['VAZAO_AR_SC_FC_CLEANER_SCAV'] = np.mean(result_df[["VAZAO_AR_SC_FC_14", "VAZAO_AR_SC_FC_15", "VAZAO_AR_SC_FC_16", "VAZAO_AR_SC_FC_17", "VAZAO_AR_SC_FC_18"]], axis=1)

    #Rougher 2
    result_df['ESPUMA_ROUGHER_FC_ROUGHER_2'] = np.mean(result_df[["ESPUMA_ROUGHER_FC_02", "ESPUMA_ROUGHER_FC_04",]], axis=1)
    result_df['VAL_DARDO_FC_ROUGHER_2'] = np.mean(result_df[["VAL_DARDO_FC_002", "VAL_DARDO_FC_004"]], axis=1)
    result_df['VAZAO_AR_ROUGHER_FC_ROUGHER_2'] = np.mean(result_df[["VAZAO_AR_ROUGHER_FC_02", "VAZAO_AR_ROUGHER_FC_03", "VAZAO_AR_ROUGHER_FC_04"]], axis=1)

    #Rougher 3
    result_df['VAZAO_AR_ROUGHER_FC_ROUGHER_3'] = np.mean(result_df[["VAZAO_AR_ROUGHER_FC_05", "VAZAO_AR_ROUGHER_FC_06"]], axis=1)

    return result_df


def get_random_indexes(length, indexes):

    result = [index for index in indexes if index < length]

    return result


def split_data_random(data: pd.DataFrame, random_indexes: list,
                      test_size: float = 0.2) -> pd.DataFrame:
    """
    Split the data randomly into training and test sets.

    Parameters:
        data (pd.DataFrame): Full dataset.
        test_size (float, optional): Size of the test set. Defaults to 0.2.

    Returns:
        tuple: Tuple containing two elements - train data and test data.
    """
    idx = get_random_indexes(data.shape[0], random_indexes['indexes'])

    train_size = int((1-test_size) * data.shape[0])

    train = data.iloc[idx[:train_size]].sort_index()
    test = data.iloc[idx[train_size:]].sort_index()

    return train, test


def split_data_datetime(data: pd.DataFrame,
                        test_size: float = 0.2) -> pd.DataFrame:
    """
    Split the data based on the specified datetime, creating training and test sets.

    Parameters:
        data (pd.DataFrame): Full dataset.
        test_size (float): Proportion of the data to include in the test split.

    Returns:
        tuple: Tuple containing three elements - train data without outliers, test data with outliers,
        and test data without outliers.
    """

    train_size = int((1-test_size) * data.shape[0])

    train = data.iloc[:train_size,]
    test = data.iloc[train_size:,]

    return train, test


def split_data(agg_data, random_indexes, params):
    """
    Split the data based on the specified method and parameters.

    Parameters:
        full_data: Full dataset.
        start_test_datetime: Datetime value to start the test set.
        remove_outliers_from (list[str]): List of models for which outliers should be removed.
        params (dict): Dictionary containing method-specific parameters.

    Returns:
        tuple: Tuple containing three elements - cleaned training data, original test data, and test data without outliers.
    """

    test_size = params['test_size']

    train_random, test_random = split_data_random(agg_data, random_indexes, test_size)
    train_datetime, test_datetime = split_data_datetime(agg_data, test_size)

    return (train_random, test_random,
            train_datetime, test_datetime)

