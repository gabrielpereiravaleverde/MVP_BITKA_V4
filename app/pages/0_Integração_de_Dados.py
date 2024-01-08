import streamlit as st

from PIL import Image
from pathlib import Path

from kedro.framework.startup import bootstrap_project

from libs.utils.utils import *
from libs.integrador import main as integrador

# ---- Page Setup ----

st.set_page_config(page_title="Integração de Dados", layout="wide")

project_path = Path(__file__).resolve().parents[2]
bootstrap_project(project_path)

image = Image.open("app/ícones/mvv.jpg")

# Crie duas colunas na página
col1, col2 = st.columns([3, 1])

# Na primeira coluna (col1), adicione o título "Sobre"
with col1:
    st.title("Módulo de Integração de Dados")

# Na segunda coluna (col2), adicione a imagem do logo
with col2:
    st.image(image, width=150)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('''
O objetivo deste módulo é atualizar a base de dados utilizada pelos outros
módulos (Simulador e Otimizador).
''')

st.markdown('### Atualizar Base de Dados Analítica do Simulador e Otimizador:',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Importação via planilha Excel",
                     "Importação via Banco de Dados"])

# ---- Importação de Dados ----

catalog = load_catalog_config()

if 'processed' not in st.session_state:
    st.session_state.processed = 0
if 'processed_except' not in st.session_state:
    st.session_state.processed_except = None
if 'updated' not in st.session_state:
    st.session_state.updated = None
if 'updated_except' not in st.session_state:
    st.session_state.updated_except = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = ''
if 'updated_date' not in st.session_state:
    st.session_state.updated_date = ''

datasets = {
    'balanco_de_massas': 'Balanço de Massas',
    'blend': 'Blend',
    'carta_controle_pims': 'Carta Controle PIMS',
    'laboratorio': 'Laboratório',
    'laboratorio_raiox': 'Laboratorio Raio X',
    'reagentes': 'Reagentes'
}

with tab1:
    for key, value in datasets.items():
        if key not in st.session_state:
            st.session_state[key] = None
        integrador.read_excel(key, value, st.session_state)
        if key in st.session_state and st.session_state[key] is not None:
            df = st.session_state[key]
            st.success(f"Dados de **{value}** carregados com **sucesso**!")

    st.button('Processar Dados de Entrada', key='xlsx_process', on_click=integrador.kedro_run, args=[
        'web_app', ['web_app'], catalog, st.session_state], type='secondary')

    if (st.session_state.processed == 1):
        st.exception(st.session_state.processed_except)
        st.session_state.processed_except = None
    elif (st.session_state.processed not in [0, 1]):
        st.success(f"**Dados de Entrada** processados com **sucesso**!")
        st.info(f"Informação mais recente é de **{st.session_state.last_update}**.")

    st.button('Atualizar Base de Dados', key='xlsx_update',
              disabled=False if st.session_state.processed == 2 else True,
              on_click=integrador.replace_database,
              args=['merged_raw_data', 'hourly_data', catalog, st.session_state])
    if st.session_state.updated:
        st.success(
            f'Base de Dados Analítica do Simulador e do Otimizador atualizada com sucesso em {st.session_state.updated_date}!')
    elif st.session_state.updated == False:
        st.exception(st.session_state.updated_except)

with tab2:

    file = st.file_uploader(
        "Selecione a planilha de **Reagentes**:", type='xlsx', key=f'reagentes_sql', disabled=True)

    st.button('Processar Dados de Entrada', key='sql_process', on_click=integrador.kedro_run, args=[
        'web_app', ['web_app'], catalog, st.session_state], type='secondary', disabled=True)

    st.button('Atualizar Base de Dados', key='sql_update', disabled=True,
              on_click=integrador.replace_database,
              args=['merged_raw_data', 'hourly_data', catalog])
