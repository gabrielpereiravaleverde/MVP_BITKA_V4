import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from libs.utils.streamlit_integration.widgets import *
from libs.utils.business_logic.data_processing import *
from typing import Dict, List, Any, Tuple


def render_tabs(X: pd.DataFrame, y: Any, etapas: Dict[str, Dict[str, list]], key: str, model) -> None:
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
    selected_day_data = None

    st.markdown("<h2 style='text-align: left; font-size: 20px;'>Buscar Dados de Entrada:</h2>",
                unsafe_allow_html=True)

    if 'selected_date' not in st.session_state or not (X.index[0].date() <= st.session_state['selected_date'] <= X.index[-1].date()):
        st.session_state['selected_date'] = X.index[0].date()
    if 'load_dataset_by_date_hour' not in st.session_state:
        st.session_state['load_dataset_by_date_hour'] = False

    tab1, tab2 = st.tabs(["Seleção de Data e Hora do Histórico",
                         "Importação de Dados via Planilha Excel"])

    with tab1:
        if 'selected_date' not in st.session_state:
            st.session_state['selected_date'] = X.index[0].date()
        if 'selected_date' not in st.session_state:
            st.session_state['selected_date'] = X.index[0].date()

        def update_selected_date():
            st.session_state['selected_date'] = st.session_state['selected_date_input']

        if 'selected_time_str' not in st.session_state:
            st.session_state['selected_time_str'] = "00:00:00"  # Valor inicial

        def load_last_datetime():
            last_datetime = X.index[-1]
            last_date = last_datetime.date()
            last_time = last_datetime.time().strftime('%H:%M:%S')

            st.session_state['selected_date'] = last_date
            st.session_state['selected_time_str'] = last_time
            selected_datetime = pd.to_datetime(f"{last_date} {last_time}")
            lag_1hr = selected_datetime - pd.Timedelta(hours=1)
            lag_2hr = selected_datetime - pd.Timedelta(hours=2)

            if lag_1hr in X.index and lag_2hr in X.index:
                selected_day_data = X.loc[selected_datetime]
                lag1_data = X.loc[lag_1hr]
                lag2_data = X.loc[lag_2hr]
                combined_data = pd.concat([selected_day_data, lag1_data, lag2_data], axis=1)
                combined_data.columns = ['Selected', 'Lag_1hr', 'Lag_2hr']
                st.session_state['combined_data'] = combined_data
                st.session_state['load_dataset_by_date_hour'] = True

                on_button_click(selected_time_str = last_time)

        with st.container():
            st.button('Selecionar Última Data e Hora', on_click=load_last_datetime)


            selected_date_input = st.date_input(
                "Selecione uma data:",
                key='selected_date_input',  # Use a key for the widget
                value=st.session_state['selected_date'],
                min_value=X.index[0].date(),
                max_value=X.index[-1].date(),
                on_change=update_selected_date
            )

        available_hours = X.index[X.index.date == st.session_state['selected_date']]

        if hours_str_list := [time.strftime('%H:%M:%S') for time in available_hours.time]:
            selected_time_str = st.selectbox("Selecione uma hora:", hours_str_list, index=hours_str_list.index(st.session_state['selected_time_str']))
            selected_datetime = pd.to_datetime(
                f"{selected_date_input} {selected_time_str}")

            # Retrieve lagged data
            lag_1hr = selected_datetime - pd.Timedelta(hours=1)
            lag_2hr = selected_datetime - pd.Timedelta(hours=2)

            if lag_1hr in X.index and lag_2hr in X.index:
                selected_day_data = X.loc[selected_datetime]
                lag1_data = X.loc[lag_1hr]
                lag2_data = X.loc[lag_2hr]

                # Combine the selected and lagged data
                try:
                    combined_data = pd.concat(
                        [selected_day_data, lag1_data, lag2_data], axis=1)
                    combined_data.columns = ['Selected', 'Lag_1hr', 'Lag_2hr']
                    st.session_state['load_dataset_by_date_hour'] = True
                except pd.errors.InvalidIndexError as e:
                    st.session_state['load_dataset_by_date_hour'] = False
                    msg = ('A base de dados possui entradas duplicadas, '
                           'não foi possível selecionar uma data e hora. '
                           'Tente outra combinação de data e hora.')
                    st.error(msg)
            else:
                st.warning(
                    "Não há dados de lag disponíveis para a data e hora selecionadas.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.index, y=y, mode='lines+markers',
                      line=dict(color='#247454'), name='Y'))
        fig.update_xaxes(title_text='Data')
        fig.update_yaxes(title_text='Teor de Cobre na CD ao longo do tempo')
        fig.update_layout(width=1200, height=400)
        st.plotly_chart(fig)

        def on_button_click(selected_time_str):
            active_tab = st.session_state.get(
                "active_tab", "Seleção de Data e Hora do Histórico")
            if active_tab == "Seleção de Data e Hora do Histórico" and selected_day_data is not None:
                new_input_values = {}
                for col in combined_data.columns:
                    new_input_values[col] = combined_data[col].to_dict()

                st.session_state[key] = new_input_values
                st.session_state['selected_time_str'] = selected_time_str

        st.button("Carregar Dados de Entrada a partir de Data e Hora",
                  disabled=not st.session_state['load_dataset_by_date_hour'], on_click=on_button_click(selected_time_str = selected_time_str))
        st.markdown("<hr>", unsafe_allow_html=True)

    with tab2:
        feature_names = model.explain_global().columns
        suffixes = ['_min', '_max', '_median', '_diff_min_max']
        filtered_feature_names = filter_feature_names(feature_names, suffixes)

        if st.button("Gerar link para download de planilha com dados de entrada"):
            excel_file = create_excel_file(filtered_feature_names)
            download_link = get_excel_download_link(excel_file, "template.xlsx")
            st.markdown(download_link, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Adicione a planilha com os dados", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            uploaded_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(
                '.csv') else pd.read_excel(uploaded_file)

        if st.button("Carregar Dados de Entrada a partir de Planilha"):
            if uploaded_file is None:
                st.error(
                    "Erro: Nenhuma planilha foi carregada. Por favor, carregue um arquivo.")
            else:
                uploaded_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(
                    '.csv') else pd.read_excel(uploaded_file)
                erro = False
                non_numeric_values = []

                # Variáveis esperadas que estão faltando
                missing_vars = set(filtered_feature_names) - \
                    set(uploaded_data["Variáveis"])
                if missing_vars:
                    st.error(
                        f"Erro: As seguintes variáveis esperadas estão faltando na planilha carregada: {', '.join(missing_vars)}")
                    erro = True

            for col in ["No Horário", "1 Hora atrás", "2 Horas atrás"]:
                for i, row in uploaded_data.iterrows():
                    try:
                        pd.to_numeric(row[col])
                    except ValueError:
                        non_numeric_values.append((row["Variáveis"], col))

            if non_numeric_values:
                st.error("Erro: Os seguintes valores não são numéricos: " +
                         ", ".join([f"{var} em '{col}'" for var, col in non_numeric_values]))
                erro = True

            if not erro:
                # Initialize new structure for input values
                new_input_values = {"Selected": {}, "Lag_1hr": {}, "Lag_2hr": {}}
                for _, row in uploaded_data.iterrows():
                    variable = row["Variáveis"]
                    new_input_values["Selected"][variable] = row["No Horário"]
                    new_input_values["Lag_1hr"][variable] = row["1 Hora atrás"]
                    new_input_values["Lag_2hr"][variable] = row["2 Horas atrás"]

                st.session_state[key] = new_input_values

    st.markdown("<h2 style='text-align: left; font-size: 20px;'>Modificar Dados de Entrada:</h2>",
                unsafe_allow_html=True)
    render_inputs_for_stage_and_variable_type(
        etapas, key, st.session_state.get(key, {}))

    def update_template_with_input_values(template_df: pd.DataFrame, input_values: Dict[str, Dict[str, Any]]) -> pd.DataFrame:

        for var in template_df["Variáveis"]:
            for lag, col_name in zip(['Selected', 'Lag_1hr', 'Lag_2hr'], ["No Horário", "1 Hora atrás", "2 Horas atrás"]):
                if lag in input_values and var in input_values[lag]:
                    value_to_insert = input_values[lag][var]
                    template_df.loc[template_df["Variáveis"]
                                    == var, col_name] = value_to_insert
                else:
                    # Para debugging
                    print(
                        f"Variável {var} ou etapa {lag} não encontrada em input_values")

        return template_df

    def add_download_button_for_template(template_df: pd.DataFrame) -> io.BytesIO:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            template_df.to_excel(writer, index=False)
        output.seek(0)
        return output

    def prepare_and_update_template(model, key: str, input_values: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        feature_names = model.explain_global().columns
        suffixes = ['_min', '_max', '_median', '_diff_min_max']
        filtered_feature_names = filter_feature_names(feature_names, suffixes)

        columns = ["Variáveis", "No Horário", "1 Hora atrás", "2 Horas atrás"]
        template_data = {col: [] for col in columns}
        template_data["Variáveis"] = filtered_feature_names
        for col in columns[1:]:
            template_data[col] = [""] * len(filtered_feature_names)
        template_df = pd.DataFrame(template_data)

        # Atualiza o DataFrame com os valores de entrada atuais
        updated_template_df = update_template_with_input_values(
            template_df, input_values)

        return updated_template_df

    if st.button("Gerar link para download de template de planilha com dados de entrada"):
        # Obtém os valores de entrada atuais
        current_input_values = st.session_state.get(
            key, {"Selected": {}, "Lag_1hr": {}, "Lag_2hr": {}})

        # Prepara e atualiza o DataFrame do template
        updated_template_df = prepare_and_update_template(
            model, key, current_input_values)

        # Gera o arquivo Excel e obtém o link de download
        excel_file = add_download_button_for_template(updated_template_df)
        download_link = get_excel_download_link(excel_file, "dados_de_entrada.xlsx")
        st.markdown(download_link, unsafe_allow_html=True)
