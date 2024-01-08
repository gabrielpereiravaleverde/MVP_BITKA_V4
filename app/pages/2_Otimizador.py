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
from src.mvvflotacao.shared_code.modeling import ConformalModel, Ensemble, Explaining

# Configurações da página
st.set_page_config(
    page_title="Otimizador",
    layout='wide',
    page_icon=":gear:",
)

catalog = load_catalog_config()

data = catalog.load("hourly_data")
data_copy = data[[c for c in data.columns if c != 'CONC_ROUG_FC01_CUT']].copy()
data_copy.set_index('DATA', inplace=True)
data_copy.sort_index(inplace=True)
model = catalog.load("conformal_model_conc_cd_randomly")
model_predict = catalog.load("ebm_conc_cd_randomly")
model = model['model']
model_predict = model_predict['model']

# Colunas relevantes

# Configuração da interface
image = Image.open("app/ícones/mvv.jpg")
col1, col2 = st.columns([3, 1])

with col1:
    st.title("Otimizador da Flotação do Concentrado de Cobre na CD")

with col2:
    st.image(image, width=150)

st.markdown("<hr>", unsafe_allow_html=True)

initialize_state()

model_variables = model.explain_global().columns

with open('app/data/00_metadata/etapas.yaml', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)

etapas = generate_variable_dict(model_variables, yaml_data, include_all=True)

result_dict = generate_variable_dict(model_variables, yaml_data, include_all=False)

decision_variables = []
for category in result_dict.values():
    for subcategory in category.values():
        for var_info in subcategory:
            decision_variables.append(var_info[0])  # Adiciona apenas o nome da variável

relevant_cols = [col for col in data.columns if col in decision_variables and not re.search(r'_lag\d+$', col) and not col.endswith('_sqr2')]

target_copy = data['CONC_ROUG_FC01_CUT'].copy()
input_values_optimizer = render_tabs(data_copy, target_copy, etapas, 'input_values_optimizer', model)

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
    output_variables = [col for col in data.columns if col in decision_variables and not re.search(r'_lag\d+$', col) and not col.endswith('_sqr2')]
    default_limits = (0.0, 30.0)
    user_defined_restrictions = render_restriction_inputs(etapas, decision_variables, default_limits, data)

    
recup_metal = None

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; font-size: 30px;'>Otimização</h1>",
    unsafe_allow_html=True,
)

etapas = generate_variable_dict(model_variables, yaml_data, include_all=False)

if 'input_values_optimizer' in st.session_state and st.session_state['input_values_optimizer'] == st.session_state['initial_input_values_optimizer']:
    st.warning("Nenhuma variável foi alterada.")

type_opt = yaml_data['otimizador'][0][0]

st.button("Otimizar", on_click = on_optimize_button_click, args = (type_opt, model, model_predict, copy.deepcopy(st.session_state.input_values_optimizer), data, copy.deepcopy(user_defined_restrictions), relevant_cols.copy()))

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
                                            st.session_state.cu_teor,
                                            model_variables)

    excel_data = to_excel_download_link(results_df)

    # Botão para download direto dos resultados
    st.download_button(label="Exportar Resultados",
                    data=excel_data,
                    file_name="optimized_data.xlsx",
                    mime="application/vnd.ms-excel")

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

filtered_etapas = {key: value for key, value in etapas.items() if value}

