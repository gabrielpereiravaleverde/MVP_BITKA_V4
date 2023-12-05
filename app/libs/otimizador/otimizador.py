
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, List
import libs.otimizador.optimizer as opt
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
    st.text_input("Teor de Cu Estimado na CD (%):", value=str(st.session_state.cu_teor) if st.session_state.cu_teor else "")
    st.text_input("Recuperação Metalúrgica Estimada na CD (%):", value=str(recup_metal) if recup_metal else "", disabled=True)

def create_optimization_results_df(
    input_values: Dict[str, float], 
    optimization_results: Dict[str, float], 
    restrictions: Dict[str, float], 
    conc_roug_fc01_cut: float
) -> pd.DataFrame:
    """
    Create a DataFrame summarizing the optimization results, including the optimized values and restrictions.

    Args:
        input_values (dict): The initial input values for the optimization process.
        optimization_results (dict): The results of the optimization process as a dictionary of variable names and their optimized values.
        restrictions (dict): The restrictions applied during the optimization process, as a dictionary of variable names and their restriction values.
        conc_roug_fc01_cut (float): The final value of the 'CONC_ROUG_FC01_CUT' variable after optimization.

    Returns:
        pd.DataFrame: A DataFrame that combines both the optimized variable values and the restrictions in a structured format for easy visualization and analysis.

    The DataFrame is created by concatenating two separate DataFrames: one for the optimized values and one for the restrictions.
    """
    # Resultados da Otimização e valor da variável alvo
    optimized_values_data = {"Variáveis": list(optimization_results.keys()) + ["CONC_ROUG_FC01_CUT"], 
                             "Valores": list(optimization_results.values()) + [conc_roug_fc01_cut]}

    # Restrições
    restrictions_data = {"Variáveis": [f"Restrição - {k}" for k in restrictions.keys()], 
                         "Valores": list(restrictions.values())}

    # Concatenar os dois DataFrames
    df_optimized = pd.DataFrame(optimized_values_data)
    df_restrictions = pd.DataFrame(restrictions_data)
    results_df = pd.concat([df_optimized, df_restrictions], ignore_index=True)

    return results_df
