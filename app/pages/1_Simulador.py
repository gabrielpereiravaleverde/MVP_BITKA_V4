import os
import yaml
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from kedro.io import DataCatalog
from typing import Dict, List, Any, Tuple
import time
from PIL import Image
from interpret.glassbox import ExplainableBoostingRegressor
from libs.simulador.simulador import *
from libs.utils.utils import *
from src.mvvflotacao.shared_code.modeling import ConformalModel, Ensemble, Explaining

# Agora, etapas contém o dicionário necessário
st.set_page_config(
    page_title="Simulador",
    layout='wide',
    page_icon="🎲",
)

catalog = load_catalog_config()

image = Image.open("app/ícones/mvv.jpg")

# Crie duas colunas na página
col1, col2 = st.columns([3, 1])

# Na primeira coluna (col1), adicione o título "Sobre"
with col1:
    st.title("Simulador da Flotação do Concentrado de Cobre na CD")

# Na segunda coluna (col2), adicione a imagem do logo
with col2:
    st.image(image, width=150)


st.markdown("<hr>", unsafe_allow_html=True)

# Start Streamlit app
option = st.selectbox(
    "Escolha o tipo de Análise:",
    ["Teor de Cobre na CD"]
)

st.markdown(f"<h1 style='text-align: center; font-size: 25px;'>Análise Preditiva {option}</h1>", unsafe_allow_html=True)

if option == "Teor de Cobre na CD":
    data = catalog.load("hourly_data")
    model = catalog.load("conformal_model_conc_cd_randomly")
    model = model['model']
    X = data[[c for c in data.columns if c != 'CONC_ROUG_FC01_CUT']].copy()
    X.set_index('DATA', inplace=True)
    X.sort_index(inplace=True)
    y = data['CONC_ROUG_FC01_CUT'].copy()
for col in X.columns:
    if pd.api.types.is_float_dtype(X[col]):
        X[col] = X[col].astype('float64')
model_variables = model.explain_global().columns

with open('app/data/00_metadata/etapas.yaml', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)

etapas = generate_variable_dict(model_variables, yaml_data, include_all=True)

stage_keys = list(etapas.keys())

input_values_simulator = render_tabs(X, y, etapas, 'input_values_simulator', model)

def add_interaction_variables(model_variables, etapas_dict):
    interaction_vars = [var for var in model_variables if '&' in var]

    # Verifica se existem variáveis de interação
    if interaction_vars:
        # Adicionar cada variável de interação como uma tupla (nome, "In")
        interaction_tuples = [(var, "In") for var in interaction_vars]
        etapas_dict.setdefault('Interações entre Variáveis', {}).setdefault('Interações', []).extend(interaction_tuples)

    return etapas_dict

# Primeiro, gerar o dicionário com as variáveis existentes
etapas = generate_variable_dict(model_variables, yaml_data, include_all=True)

# Em seguida, adicionar as variáveis de interação
etapas = add_interaction_variables(model_variables, etapas)

stage_keys = list(etapas.keys())
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; font-size: 25px;'>Simulação</h1>",
    unsafe_allow_html=True,
)

if 'simulated' not in st.session_state:
    st.session_state.simulated = False
if 'selected_stage' not in st.session_state:
    st.session_state.selected_stage = "Todas"
if 'last_fig' not in st.session_state:
    st.session_state.last_fig = None
if 'predicted_value' not in st.session_state:
    st.session_state.predicted_value = None
    
all_options = ["Todas"] + stage_keys

if st.button("Simular", on_click=lambda: simulate(
    st.session_state.input_values_simulator, 
    st.session_state.selected_stage, 
    model, 
    y, 
    etapas,
    st.session_state)):
    pass

if st.session_state.simulated:
    st.markdown(f"**O valor estimado para Teor de cobre no Concentrado da CD é:** {st.session_state.predicted_value[0].round(2)}%")
    st.markdown(f"**O intervalo da predição é:** [{st.session_state.predicted_value[1].round(2)}%, {st.session_state.predicted_value[2].round(2)}%]")
    st.markdown('**Interpretação do impacto de cada variável.**')

    # Stage selection after simulation
    selected_visualization_stage = st.selectbox("Selecione a Etapa para Visualização", all_options)
        # Check if a stage is selected and update the plot
    if selected_visualization_stage:
        st.session_state.last_fig = update_visualization(
            selected_visualization_stage, 
            st.session_state.current_values, 
            model, 
            y, 
            etapas
        )
        # Display the plot with the values for the selected stage
    if st.session_state.last_fig is not None:
        st.plotly_chart(st.session_state.last_fig)
