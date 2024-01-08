# Pipeline Otimização 

`kedro run --pipeline optimization`

## Inputs
* Modelo treinado: ebm_conc_cd
* Dados completos: conc_cd_full_data_without_correlation_s0

## Outputs
* Plots análises salvo em data/08_reporting: conc_cd_simulations_plot 

## Parâmetros (parameters_optimization.yml)
* gs_steps: quantidade de valores utilizados a serem testados no gridsearch
* bayesian_max_time: tempo total de execução em minutos da otimização bayesiana
* bayesian_max_it: quantidade máxima de iterações da otimização bayesiana
* decision_variables: lista de variáveis de decisão para serem otimizadas
* scenario_features: features adicionais (além da variável target e das variáveis de decisão) contidas em conc_cd_full_data_without_correlation_s0 que são utilizadas para escolher cenários distintos a serem testados