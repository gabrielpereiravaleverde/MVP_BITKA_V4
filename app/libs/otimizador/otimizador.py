
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, List
import libs.otimizador.optimization.optimizer as opt
from libs.otimizador.business_logic import *
from libs.otimizador.streamlit_integration import *

def render_optimization_results(stages: Dict[str, Dict[str, List[Tuple[str, str]]]], best_values: Dict[str, float], current_values: pd.Series, recup_metal) -> None:
    st.markdown(
        "<h2 style='text-align: left; font-size: 20px;'>Set Points sugeridos pelo Otimizador:</h2>",
        unsafe_allow_html=True,
    )
    current_values = pd.DataFrame(current_values, index = [0])

    stage_tabs = st.tabs(list(stages.keys()))
    for stage in stages:
        with stage_tabs[list(stages.keys()).index(stage)]:
            for variable_type, variable_tuples in stages[stage].items():
                for variable_tuple in variable_tuples:
                    if isinstance(variable_tuple, tuple) and len(variable_tuple) == 2:
                        variable, _ = variable_tuple
                    else:
                        continue  # Se não for uma tupla válida, pula esta variável

                    if variable in best_values:
                        st.text_input(f"{display_name(variable_tuple)} - Valor Proposto:", value=str(best_values[variable]), key=f"result_{variable}")
                        if variable in current_values:
                            st.text_input(f"{display_name(variable_tuple)} - Valor Anterior:", value=str(current_values[variable].iloc[0].round(2)), key=f"current_{variable}", disabled=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align: left; font-size: 20px;'>Resultado Estimado com o uso dos Set Points sugeridos pelo Otimizador:</h2>",
        unsafe_allow_html=True,
    )
    value = str(st.session_state.cu_teor[0]) if st.session_state.cu_teor.size > 0 else 'No value'
    st.text_input("Teor de Cu Estimado na CD (%):", value=value)
    if st.session_state.cu_teor.size > 2:
        value_interval = f"[{str(st.session_state.cu_teor[1])}, {str(st.session_state.cu_teor[2])}]"
    else:
        value_interval = "No values"
    st.text_input("Intervalo da predição (%):", value=value_interval)
    st.text_input("Recuperação Metalúrgica Estimada na CD (%):", value=str(recup_metal) if recup_metal else "", disabled=True)

def create_optimization_results_df(
    input_values: Dict[str, Dict[str, float]],  # Nested dictionary for input values
    optimization_results: Dict[str, float],
    restrictions: Dict[str, float],
    conc_roug_fc01_cut: float,
    model_variables: List[str]  # Variables considered in the model
) -> pd.DataFrame:
    """
    Create a DataFrame summarizing the optimization results in multiple sections.
    """
    filtered_input_values = {}
    for time_key, time_values in input_values.items():
        filtered_input_values[time_key] = {var: val for var, val in time_values.items() if var in model_variables}

    # Processamento dos Valores de Entrada
    input_values_flat = []
    for var, time_values in filtered_input_values.get("Selected", {}).items():
        row = [var] + [time_values, filtered_input_values.get("Lag_1hr", {}).get(var, None), filtered_input_values.get("Lag_2hr", {}).get(var, None)]
        input_values_flat.append(row)
    
    df_input_values = pd.DataFrame(input_values_flat, columns=["Variáveis", "No Horário", "1 Hora atrás", "2 Horas atrás"])

    # Processamento das Restrições
    df_restrictions = pd.DataFrame({
        "Variáveis": restrictions.keys(),
        "Restrições": restrictions.values()
    })

    # Adicione o valor recomendado da otimização nas variáveis que têm restrições
    df_restrictions["Valores Recomendados"] = df_restrictions["Variáveis"].apply(lambda var: optimization_results.get(var, None))

    # Processamento do Intervalo de Resultados
    df_final_result = pd.DataFrame({
        "Variáveis": ["CONC_ROUG_FC01_CUT"],
        "Intervalo de Resultados": [conc_roug_fc01_cut]
    })

    # Concatenando todas as seções
    results_df = pd.concat([df_input_values, df_restrictions, df_final_result], ignore_index=True)

    return results_df