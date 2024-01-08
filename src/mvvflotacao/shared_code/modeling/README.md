# Modelagem 

Nesta pasta, estão os principais arquivos relacionados aos modelos do pipeline. 

O overview do passo-a-passo para se desenvolver novos modelos é:

1. Métodos de Explicabilidade em Explain.py e implementação da interface padrão de modelos
2. Modelo deve possuir o método get_features(). Se for um modelo importado de pacote, pode-se fazer um extension no arquivo __init__.py dessa pasta
3. O arquivo shared_code.utils deve suportar a construção desses novos modelos


## 1. Métodos de Explicabilidade: 

Todo modelo deve suportar a execução de dois métodos `explain_local(X)` e `explain_global()`. Para modelos custom, onde você desenvolve uma nova classe (exemplo é `shared_code.modeling.Ensemble`), deve-se herdar a interface
`shared_code.modeling.Interface` e implementar os métodos `fit(.)`, `predict(.)`, `get_features(.)`, `exaplain_local(.)` e `explain_global(.)`, 


Para modelos já desenvolvidos, ou seja, algum modelo que se importa de um pacote pronto, por exemplo um simples LinearRegression, deve-se implementar os métodos de explicabilidade em `shared_code.modeling.Explain`, ou seja,
os dicionários `__local_exp` e `__local_global` devem possuir respectivamente funções para gerar o `exaplain_local(.)` e `explain_global(.)` desse modelo. 

Deve-se seguir o padrão e mesma estrutura de outputs que são Pandas Dataframes. Verificar a classe `shared_code.modeling.Explain` para se checar o padrão. 


## 2. GetFeatures:

O modelo deve suportar a execução do método `get_features(.)`, deve-se herdar a interface `shared_code.modeling.Interface` e implementar os métodos `fit(.)`. 

Para modelos já desenvolvidos, ou seja, algum modelo que se importa de um pacote pronto, por exemplo um simples LinearRegression, deve-se implementar este método como extension method em `shared_code.modeling.__init__`. 

Checar este arquivo para se ver um exemplo de como foi feito para o modelo Catboost.

## 3. Suporte para Construção do Modelo

O arquivo `shared_code.utils` deve suportar a construção desse tipo de modelo, inserindo-o no dicionário `supported_models`. A chave neste dicionário deve ser padronizada como que vem do arquivo de configuração `data_science.yml`.

