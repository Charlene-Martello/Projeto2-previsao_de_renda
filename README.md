# Projeto de Previsão de Renda

Este projeto tem como objetivo criar um modelo preditivo para estimar a renda de indivíduos com base em diversas variáveis socioeconômicas. Para isso, foram desenvolvidos dois notebooks: um focado na construção do modelo preditivo e análise dos resultados, e outro focado na visualização gráfica interativa dos dados e dos resultados.

## Estrutura do Projeto

### 1. [Notebook previsão de renda]
Este notebook contém a análise detalhada e a criação do modelo preditivo para a previsão de renda. Aqui, utilizamos diversas técnicas de aprendizado de máquina para construir um modelo robusto, sendo o processo dividido nas seguintes etapas:

- **Carregamento dos dados:** utilização do comando pd.read_csv.
- **Análise Exploratória de Dados (EDA)**: Iniciamos essa análise utilizando o ProfileReport para ter acesso a um relatório geral das variáveis contidas no nosso df. Em seguida criamos alguns gráficos para analise bivariada, a fim de identificar variáveis com possível potencial de predição da variável alvo, essa análise foi feita no notebook princial e também utilizando o Streamlit.
- **Preparação dos Dados**: Tratamento de dados faltantes.
- **Treinamento e Avaliação de Modelos**: Foram testados os modelos de regressão linear e árvores de regressão. O modelo final foi escolhido com base em métricas como o R², MAE (Mean Absolute Error) e RMSE (Root Mean Squared Error).
- **Principais Resultados**: As variáveis mais importantes para a previsão de renda incluem tempo de emprego e idade. 


### 2. [Análise Gráfica Interativa]
Este notebook é focado na visualização gráfica interativa dos dados e dos resultados obtidos. Ele permite que o usuário explore diferentes aspectos da análise, tais como:

- **Distribuição das variáveis**: Gráficos interativos que mostram como as variáveis estão distribuídas, permitindo visualizar padrões nos dados.
- **Correlação entre as variáveis**: Gráficos que ajudam a entender as relações entre as variáveis independentes e a variável alvo.

## Conclusão

Este projeto fornece uma visão completa do processo de criação de um modelo preditivo de renda, desde a análise dos dados até a visualização dos resultados. O notebook principal detalha todas as etapas envolvidas na construção do modelo, enquanto o notebook de análise gráfica interativa facilita a exploração visual dos dados e das previsões.
