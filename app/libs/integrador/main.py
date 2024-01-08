import pandas as pd
import streamlit as st

from datetime import datetime

from kedro.framework.session import KedroSession


def kedro_run(env, tags, catalog, session_state):
    try:
        with KedroSession.create(env=env) as session:
            session.run(tags=tags)
            st.session_state.last_update = catalog.load(
                'merged_raw_data')['DATA'].max().strftime('%d/%m/%Y %H:%M')
        session_state.processed = 2
    except Exception as e:
        session_state.processed_except = e
        session_state.processed = 1


def replace_database(input_name, output_name, catalog, session_state):
    try:
        df = catalog.load(input_name)
        df.to_parquet(f'./app/data/03_primary/{output_name}.pq')
        session_state.updated = True
        session_state.updated_date = datetime.today().strftime("%d/%m/%Y %H:%M")
    except Exception as e:
        session_state.updated = False
        session_state.updated_except = e


def read_excel(name, label, session_state):
    file = st.file_uploader(
        f"Selecione a planilha de **{label}**:", type='xlsx', key=f'{name}_xlsx')
    if file is not None and session_state[name] is None:
        df = pd.read_excel(file)
        df.to_excel(f'./app/data/01_raw_data/{name}.xlsx', index=False)
        session_state[name] = df.head(50)
