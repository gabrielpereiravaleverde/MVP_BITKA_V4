01 - Baixe e instale o anaconda através do link:
https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Windows-x86_64.exe

02 - Após instalado, procure pelo Anaconda Prompt na barra de pesquisa do windows e digite o seguinte comando para criar o ambiente virtual:
conda create --name sim_flot python=3.10

03 - Ative o ambiente virtual com o seguinte comando:
conda activate sim_flot

04 - Utilizando o comando "cd", acesse a pasta do sim_flot pelo conda prompt.

05 - Instale as bibliotecas com o comando:
pip install -r src/requirements.txt

06 - Inicialize o servidor web com o comando abaixo. Após essa etapa, a aplicação poderá ser acessada pelo navegador:
streamlit run app/Sobre.py


Obs: Uma vez realizado todo o processo, para inicializar a aplicação novamente, basta apenas realizar os passos 03, 04 e 06.
Obs 2: Para utilizar o Simulador e o Otimizador simultaneamente, é possível abrir duas abas ao mesmo tempo com o endereço do WebApp.