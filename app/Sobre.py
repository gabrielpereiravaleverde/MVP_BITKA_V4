import os
import yaml
import streamlit as st
from kedro.io import DataCatalog
import time
from PIL import Image
from pathlib import Path 
import sys 

sys.path.append(str(Path("./").resolve()))
print(sys.path)

# Constants
CURRENT_FILE_PATH = os.path.abspath(__file__)
BASE_PATH = os.path.dirname(CURRENT_FILE_PATH)
CATALOG_PATH = os.path.join(BASE_PATH, "catalog.yml")

st.set_page_config(
    page_title="Sobre",
    page_icon=":information_source:"
)

image = Image.open("app/ícones/mvv.jpg")

# Create two columns on the page
col1, col2 = st.columns([3, 1])

# In the first column (col1), add the title "Sobre"
with col1:
    st.title("Simulador e Otimizador da Flotação do Concentrado de Cobre")

# In the second column (col2), add the logo image
with col2:
    st.image(image, width=150)
    
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    """
**Versão:** **MVP 0.4.0**
"""
)