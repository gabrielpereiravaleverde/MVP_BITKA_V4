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
    },
    "Interações de Variáveis": {
        "Interações": [
        ]
    }
}

st.set_page_config(
    page_title="Simulador",
    page_icon="🎲",
)

catalog = load_catalog_config()

image = Image.open("app/ícones/mvv.jpg")

# Crie duas colunas na página
col1, col2 = st.columns([3, 1])

# Na primeira coluna (col1), adicione o título "Sobre"
with col1:
    st.title("Simulador da Flotação do Concentrado de Cobre")

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
    data = catalog.load("conc_cd_full_data")
    model = catalog.load("ebm_conc_cd")
    X = data[[c for c in data.columns if c != 'target']].copy()
    y = data['target'].copy()
    
base_prediction, local_importances = calculate_impacts(X[:1], model, y)
interaction_variables = [name for name in local_importances if '&' in name]
etapas["Interações de Variáveis"]["Interações"] = [(var, '') for var in interaction_variables]

stage_keys = list(etapas.keys())

input_values_simulator = render_tabs(X, y, etapas, 'input_values_simulator')

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
    st.markdown(f"**O valor estimado para Teor de cobre no Concentrado da CD é:** {st.session_state.predicted_value.round(2)}%")
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
