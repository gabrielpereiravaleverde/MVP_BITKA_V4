import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from libs.utils.streamlit_integration.widgets import *
from libs.utils.business_logic.data_processing import *
from typing import Dict, List, Any, Tuple

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

def render_tabs(X: pd.DataFrame, y: Any, etapas: Dict[str, Dict[str, list]], key: str) -> None:
    """
    Render a tabbed interface in Streamlit for data input and visualization.

    This function creates a tabbed interface with two main tabs: one for selecting 
    historical data based on date and time, and another for importing data via an Excel 
    spreadsheet. It also renders interactive inputs for different stages and variable 
    types, and a download button for a template Excel file.

    Parameters:
    X (pandas.DataFrame): DataFrame containing the historical data used for selection.
    y (pandas.Series or similar): Series or similar object representing a dependent 
                                  variable for visualization.
    etapas (dict): A dictionary defining stages and associated variables for rendering 
                   input fields.
    key (str): A key for storing and accessing user input values in the Streamlit 
               session state.

    Notes:
    - The function uses Plotly for data visualization.
    - It includes checks for missing or extra variables, and non-numeric values in 
      the uploaded data.
    - The function uses the 'key' parameter to store and manage user inputs across 
      different tabs and sessions.
    - It dynamically updates a template DataFrame with user inputs for download.
    """    
    st.markdown("<h2 style='text-align: left; font-size: 20px;'>Buscar Dados de Entrada:</h2>", unsafe_allow_html=True)

    # Inicializa o valor de selected_date no session_state, se necessário
    if 'selected_date' not in st.session_state:
        st.session_state['selected_date'] = X.index[0].date()

    tab1, tab2 = st.tabs(["Seleção de Data e Hora do Histórico", "Importação de Dados via Planilha Excel"])

    with tab1:
        selected_date_input = st.date_input("Selecione uma data:", value=st.session_state['selected_date'], min_value=X.index[0].date(), max_value=X.index[-1].date())
        st.session_state['selected_date'] = selected_date_input
        available_hours = X.index[X.index.date == selected_date_input]

        if hours_str_list := [time.strftime('%H:%M:%S') for time in available_hours.time]:
            selected_time_str = st.selectbox("Selecione uma hora:", hours_str_list)
            selected_datetime = pd.to_datetime(f"{selected_date_input} {selected_time_str}")
            selected_day_data = X.loc[selected_datetime]
        else:
            st.warning("Não há dados disponíveis para a data selecionada.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.index, y=y, mode='lines+markers', line=dict(color='#247454'), name='Y'))
        fig.update_xaxes(title_text='Data')
        fig.update_yaxes(title_text='Teor de Cobre na CD ao longo do tempo')
        st.plotly_chart(fig)
        if st.button("Carregar Dados de Entrada a partir de Data e Hora"):
            active_tab = st.session_state.get("active_tab", "Seleção de Data e Hora do Histórico")
            if active_tab == "Seleção de Data e Hora do Histórico" and selected_day_data is not None:
                new_input_values = selected_day_data.to_dict()
                st.session_state[key] = new_input_values
                
        st.markdown("<hr>", unsafe_allow_html=True)

    with tab2:
        # Cria o DataFrame que será usado como template
        template_data = {"Variáveis": X.columns, "Valores": [""] * len(X.columns)}
        template_df = pd.DataFrame(template_data)

        # Fornece o link de download do template
        tipo = 'Template'
        st.markdown(get_table_download_link(template_df, tipo), unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Adicione a planilha com os dados", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            uploaded_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        if st.button("Carregar Dados de Entrada a partir de Planilha"):
            if uploaded_file is None:
                st.error("Erro: Nenhuma planilha foi carregada. Por favor, carregue um arquivo.")
            else:
                uploaded_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                erro = False

                # Variáveis esperadas que estão faltando
                missing_vars = set(X.columns) - set(uploaded_data["Variáveis"])
                if missing_vars:
                    st.error(f"Erro: As seguintes variáveis esperadas estão faltando na planilha carregada: {', '.join(missing_vars)}")
                    erro = True

                # Variáveis extras não esperadas
                extra_vars = set(uploaded_data["Variáveis"]) - set(X.columns)
                if extra_vars:
                    st.error(f"Erro: As seguintes variáveis extras foram encontradas na planilha carregada: {', '.join(extra_vars)}")
                    erro = True

                # Verificação de valores numéricos
                try:
                    uploaded_data["Valores"] = pd.to_numeric(uploaded_data["Valores"])
                except ValueError as e:
                    st.error(f"Erro: Alguns valores na planilha carregada não são numéricos. Detalhe do erro: {e}")
                    erro = True
                non_numeric_values = []
                for i, row in uploaded_data.iterrows():
                    try:
                        pd.to_numeric(row["Valores"])
                    except ValueError:
                        non_numeric_values.append(row["Variáveis"])

                if non_numeric_values:
                    st.error(f"Erro: Os seguintes valores não são numéricos: {', '.join(non_numeric_values)}")
                    erro = True
                if not erro:
                    new_input_values = dict(zip(uploaded_data["Variáveis"], uploaded_data["Valores"]))
                    st.session_state[key] = new_input_values

    st.markdown("<h2 style='text-align: left; font-size: 20px;'>Modificar Dados de Entrada:</h2>", unsafe_allow_html=True)
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
    }}
    render_inputs_for_stage_and_variable_type(etapas, key, st.session_state.get(key, {}))

    # Updating the template sheet with the user values
    def update_template_with_input_values(template_df: pd.DataFrame, input_values: Dict[str, Any]) -> pd.DataFrame:
        for index, row in template_df.iterrows():
            variable = row['Variáveis']
            if variable in input_values:
                template_df.at[index, 'Valores'] = input_values[variable]
        return template_df

    def add_download_button_for_template(template_df: pd.DataFrame) -> None:
        tipo = "Entrada"
        download_link = get_table_download_link(template_df, tipo)
        st.markdown(download_link, unsafe_allow_html=True)
        
    template_df = pd.DataFrame({"Variáveis": X.columns, "Valores": [""] * len(X.columns)})
    current_input_values = st.session_state.get(key, {})
    updated_template_df = update_template_with_input_values(template_df, current_input_values)
    add_download_button_for_template(updated_template_df)
