import os
import yaml
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from kedro.io import DataCatalog
from typing import Dict, Tuple, List, Any, Union
import time
from PIL import Image
import numpy as np
import re
from libs.otimizador.otimizador import *
from libs.utils.utils import *
import copy
import src.mvvflotacao.shared_code.optimization.optimizer as opt

# Definição das etapas
etapas = {
    "Moagem e Classificação": {
        "Blend": [('Sulfetado_HG', '%'), ('Sulfetado_LG', '%'), ('Sulfetado_MG', '%'), ('Sulfetado_SHG', '%')],
        "Granulometria": [('PSI_OVER_CICLO', 'μm'), ('P20_MOAGEM', 'μm'), ('P99_MOAGEM', 'μm')],
        "Hidrodinâmica": [('FLOT_AL_MASSA', 't')],
        "Característica/Condição": [('ALIM_FLOT_CU_TOT', '%')]
    },
    "Flotação Rougher 1": {
        "Regulador": [('PH_ROUGHER_COND', 'pH')],
        "Hidrodinâmica": [('VAL_DARDO_CD', '%'), ('VAZAO_AR_ROUGHER_COND', 'N.m³/h'), ('ESPUMA_ROUGHER_COND', 'mm')],
        "Termodinâmica": [('Espumante (g/t)_CD', '')],
        "Característica/Condição": [('ALIM_FLOT_PER_SOL', '%'), ('ALIM_FLOT_FE', '%'), ('ALIM_FLOT_MG', '%'), ('ALIM_FLOT_NI', '%')],
        "N/A": [('CONC_ROUG_FC01_CUT', '%')]
    }
}

# Configurações da página
st.set_page_config(
    page_title="Otimizador",
    page_icon=":gear:",
)

catalog = load_catalog_config()

data = catalog.load("conc_cd_full_data")
model = catalog.load("ebm_conc_cd")

# Colunas relevantes
relevant_cols = [col for col in data.columns if col.startswith(("VAZAO_AR_ROUGHER", "Espumante", "ESPUMA")) and not re.search(r'_lag\d+$', col) and not col.endswith('_sqr2')]

# Configuração da interface
image = Image.open("app/ícones/mvv.jpg")
col1, col2 = st.columns([3, 1])

with col1:
    st.title("Otimizador da Flotação do Concentrado de Cobre")

with col2:
    st.image(image, width=150)

st.markdown("<hr>", unsafe_allow_html=True)

initialize_state()

data_copy = data[[c for c in data.columns if c != 'target']].copy()
target_copy = data['target'].copy()
input_values_optimizer = render_tabs(data_copy, target_copy, etapas, 'input_values_optimizer')

if 'initial_input_values_optimizer' not in st.session_state:
    st.session_state['initial_input_values_optimizer'] = copy.deepcopy(input_values_optimizer)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; font-size: 25px;'>Parâmetros de Otimização</h1>",
    unsafe_allow_html=True,
)

# Tabs
tab_objetivo, tab_restricoes = st.tabs(["Objetivo", "Restrições"])

# Para visualização
desired_visual_values = {
    "CONC_ROUG_FC01_CUT": None,
    "recup_metal": None
}

with tab_objetivo:
    objective = st.radio("", ["Maximizar Teor de Cu no Concentrado do CD", "Maximizar Recuperação Metalúrgica da Flotação para o Concentrado no CD"], disabled=True)

    if objective == "Maximizar Teor de Cu no Concentrado do CD":
        desired_visual_values["recup_metal"] = st.number_input("Defina o limite mínimo de Recuperação Metalúrgica no CD:", min_value=0.0, max_value=100.0, step=0.1, disabled=True)
        objective_type = "CONC_ROUG_FC01_CUT"
    elif objective == "Maximizar Recuperação Metalúrgica no Concentrado do CD":
        desired_visual_values["CONC_ROUG_FC01_CUT"] = st.number_input("Defina o limite mínimo de Teor de Cu no CD", min_value=0.0, max_value=100.0, step=0.1, disabled=True)
        objective_type = "recup_metal"

# Seção de Restrições (Dosagem)
with tab_restricoes:
    output_variables = [col for col in data.columns if col.startswith(("VAZAO_AR_ROUGHER", "Espumante", "ESPUMA")) and not re.search(r'_lag\d+$', col) and not col.endswith('_sqr2')]
    default_limits = (0.0, 30.0)
    user_defined_restrictions = render_restriction_inputs(etapas, default_limits, data)

    
recup_metal = None

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; font-size: 30px;'>Otimização</h1>",
    unsafe_allow_html=True,
)

etapas = {
    "Flotação Rougher 1": {
        "Termodinâmica": [('Espumante (g/t)_CD', 'g/t')],
        "Hidrodinâmica": [('VAZAO_AR_ROUGHER_COND', 'N.m³/h'), ('ESPUMA_ROUGHER_COND', 'mm')],
        "N/A": [('CONC_ROUG_FC01_CUT', '%')]
    }
}

if 'input_values_optimizer' in st.session_state and st.session_state['input_values_optimizer'] == st.session_state['initial_input_values_optimizer']:
    st.warning("Nenhuma variável foi alterada.")

st.button("Otimizar", on_click = on_optimize_button_click, args = (model, copy.deepcopy(st.session_state.input_values_optimizer), data, copy.deepcopy(user_defined_restrictions), relevant_cols.copy()))

# Renderize os resultados da otimização se estiverem disponíveis
if st.session_state.get('optimization_results'):
    if st.session_state.get('optimization_results'):
    # if not st.session_state['first_run'] and st.session_state.get('optimization_results'):
        render_optimization_results(
            etapas,
            st.session_state['optimization_results'],
            st.session_state['input_values_optimizer'],
            st.session_state.get('recup_metal', None)
        )
        st.write(f"Otimização concluída em {st.session_state.get('elapsed_time', 0):.2f} segundos.")

    results_df = create_optimization_results_df(st.session_state['input_values_optimizer'], 
                                            st.session_state.get('optimization_results', {}), 
                                            user_defined_restrictions,
                                            st.session_state.cu_teor)

    excel_data = to_excel_download_link(results_df)

    # Botão para download direto dos resultados
    st.download_button(label="Exportar Resultados",
                    data=excel_data,
                    file_name="optimized_data.xlsx",
                    mime="application/vnd.ms-excel")

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

filtered_etapas = {key: value for key, value in etapas.items() if value}

