#!/usr/bin/env python
# coding: utf-8

# # Previsão de renda

# ## Etapa 1 CRISP - DM: Entendimento do negócio
# 
# Será realizada uma previsão de renda, para isso, utilizaremos uma base de dados de clientes de um determinado banco, essa base conta com diversas informações, como tempo de emprego, posse de imóvel, etc. 

# ## Etapa 2 Crisp-DM: Entendimento dos dados
# Abaixo temos um dicionário detalhado das variáveis contidas no nosso data frame. 
# 
# 
# ### Dicionário de dados
# 
# | Variável                | Descrição                                           | Tipo         |
# | ----------------------- |:---------------------------------------------------:| ------------:|
# | sexo                    | M='Masculino' F='Feminino'                          | M/F          |
# | posse_de_veiculo        | Y='Sim' N='Não'                                     | Y/N          |
# | posse_de_imovel         | Y='Sim' N='Não'                                     | Y/N          |
# | qtd_filhos              | Quantidade de filhos                                | Inteiro      |
# | tipo_renda              | Ex: Assalariado, autonômo, etc.                     | Texto        |
# | educacao                | Nível de educação (ex: secundário, superior etc)    | Texto        |
# | estado_civil            | Estado civil (ex: solteiro, casado etc)             | Texto        |
# | tipo_residencia         | Ex: casa/apartamento, com os pais etc               | Texto        |
# | idade                   | Idade em anos                                       | Inteiro      |
# | tempo_emprego           | Tempo de Emprego em Anos                            | Inteiro      |
# | qt_pessoas_residencia   | Quantidade de pessoas na residência                 | Inteiro      |
# | renda                   | Valor da renda                                   	| Racional     |
# 
# 
# 
# 

# #### Carregando os pacotes

# In[129]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import ydata_profiling #o mesmo que profilereport
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm

#arvore
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split


# #### Carregando os dados
# O comando pd.read_csv é um comando da biblioteca pandas (pd.) e carrega os dados do arquivo csv indicado para um objeto *dataframe* do pandas.

# In[130]:


renda = pd.read_csv('previsao_de_renda.csv')


# In[131]:


renda.head(2)


# #### Entendimento dos dados - Univariada
# Nesta etapa tipicamente avaliamos a distribuição de todas as variáveis. Utilizando o ProfileReport teremos acesso a um relatório geral das variáveis contidas no nosso data frame.

# In[84]:


prof = ProfileReport(renda, explorative=True, minimal=True)
prof


# In[85]:


prof.to_file('./output/renda_analisys.html')


# O relatório acima é capaz de nos trazer diversas observações, como por exemplo, a quantidade de dados faltantes em cada variável, a quantidade de zeros, etc. Lembrando que ele é apenas uma visão geral de como os dados se comportam dentro dessas variáveis.

# ### Entendimento dos dados - Bivariadas
# 
# 
# 

# A análise prévia foi realizada com ajuda do Streamlit e nos revela que algumas variáveis possuem maior potencial do que outras na hora de predizer qual será a renda do cliente. A página do Steamlit está anexada em vídeo para verificação, sendo muito importante fazê-la. Entretanto, a fim de melhorar a nossa organização, uma análise muito semelhante será realizada no presente nootebok também. 

# A seguir, realizaremos as análise das variáveis a fim de perceber sua capacidade ou não de ser preditora da variável renda:

# In[132]:


#1)renda x tipo de renda 
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='tipo_renda', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Gráfico de Linhas: Renda por Tipo de Renda ao longo do tempo')
plt.xticks(rotation=45)
plt.legend(title='Tipo de Renda')
plt.show()


# De acordo com o gráfico acima podemos perceber que Servidor Público costuma ter uma renda mais elevada, enquanto que pensionista, uma renda mais baixa.

# In[ ]:





# In[133]:


#2)renda x sexo
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='sexo', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita por Sexo')
plt.xticks(rotation=45)
plt.legend(title='Sexo')
plt.show()


# O gráfico acima torna evidente que os clientes do sexo masculino tem uma renda muito superior à do sexo feminino. 

# In[ ]:





# In[135]:


#3)renda x posse de veículo
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='posse_de_veiculo', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita por Posse de veículo')
plt.xticks(rotation=45)
plt.legend(title='Posse de Veículo')
plt.show()


# Assim como no gráfico anterior, temos valores bem discrepantes aqui, mostrando que os clientes que possuem veículos, também possuem uma renda maior. 

# In[ ]:





# In[136]:


#4)renda x posse de imovel
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='posse_de_imovel', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita por Posse de Imóvel')
plt.xticks(rotation=45)
plt.legend(title='Posse de Imóvel')
plt.show()


# O gráfico acima não revela posse de imóvel como uma variável de grande potencial de predição.

# In[ ]:





# In[138]:


#5)renda x grau de escolaridade
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='educacao', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita por Grau de Escolaridade')
plt.xticks(rotation=45)
plt.legend(title='Grau de Escolaridade')
plt.show()


# Analisando o gráfico acima pode-se dizer que pessoas com ensino superior completo tem, em média, uma renda superior aos demais. 

# In[ ]:





# In[92]:


#6)renda x estado civil
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='estado_civil', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita pelo Estado Civil')
plt.xticks(rotation=45)
plt.legend(title='Estado Civil')
plt.show()


# De acordo com o gráfico acima, o que fica nítido é que viúvos tem renda menor, enquanto que casados tem, em média, uma renda maior que os demais. 

# In[165]:


# Criando um gráfico de barras 
plt.figure(figsize=(12, 6))
sns.barplot(x='data_ref', y='renda', hue='qtd_filhos', data=renda, errorbar=None)

# Adicionando rótulos e título
plt.xlabel('Data de Referência')
plt.ylabel('Renda Total')
plt.title('Gráfico Bivariado: Renda por Tipo de Renda ao longo do tempo')

# Girando os rótulos do eixo x para melhor legibilidade
plt.xticks(rotation=45)

# Adicionando legenda
plt.legend(title='Tipo de Renda')

# Mostrando o gráfico
plt.show()


# In[ ]:





# In[ ]:





# In[93]:


#7)renda x tipo de residencia
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='tipo_residencia', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita pelo Tipo de Residência')
plt.xticks(rotation=45)
plt.legend(title='Tipo de Residência')
plt.show()


# O gráfico acima não parece ter um forte poder preditivo. 

# In[169]:





# Agora iremos realizar a análise de renda com idade, tempo de emprego, quantidade de pessoas na residência e qtd_filhos. Entretanto, as variáveis citadas possuem diversos valores diferentes, portanto, a fim de facilitar nossa observação e consequentemente entendimento, iremos agrupar esses dados de acordo com os percentis de cada variável. 

# In[140]:


#1)IDADE
quartis_idade = renda['idade'].describe(percentiles=[0.25, 0.5, 0.75])
quartis_idade


# In[96]:


# Definindo os limites para cada faixa etária
faixas_etarias = [22, 34, 43, 53, 68]
rotulos_faixas = ['22-34', '35-43', '44-53', '54+']

# Adicionando uma nova coluna 'faixa_etaria' ao DataFrame
renda['faixa_etaria'] = pd.cut(renda['idade'], bins=faixas_etarias, labels=rotulos_faixas, right=False)


# In[97]:


#renda x idade
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='faixa_etaria', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita por Faixa Etária')
plt.xticks(rotation=45)
plt.legend(title='Faixa Etária')
plt.show()


# O gráfico acima nos revela que clientes que possuem entre 44 e 53 anos possuem a maior média de renda, seguidos pelos que possuem entre 35 e 43 anos, seguidos pelos que posuem mais de 54 anos, e por fim, com a menor renda, jovens adultos que possuem entre 22 e 34 anos.

# In[ ]:





# In[142]:


#2)TEMPO EMPREGO
quartis_tempo_emprego = renda['tempo_emprego'].describe(percentiles=[0.25, 0.5, 0.75])
quartis_tempo_emprego


# In[143]:


# Definindo os limites para cada faixa de tempo de emprego
faixas_emprego = [0, 3, 6, 10, 43]
rotulos_emprego = ['0-3 anos', '3-6 anos', '6-10 anos', '10+']

# Adicionando uma nova coluna 'faixa_etaria' ao DataFrame
renda['faixas_emprego'] = pd.cut(renda['tempo_emprego'], bins=faixas_emprego, labels=rotulos_emprego, right=False)


# In[144]:


#renda x tempo de emprego
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='faixas_emprego', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita por Faixas de Tempo de Emprego')
plt.xticks(rotation=45)
plt.legend(title='Faixa Tempo de Emprego')
plt.show()


# O gráfico acima mostra tempo de emprego como uma forte preditora de renda, isso se dá porque, quanto maior o tempo de emprego, maior a renda.

# In[ ]:





# In[145]:


#3 QUANTIDADE DE PESSOAS NA RESIDENCIA 
quartis_qtd_pessoas = renda['qt_pessoas_residencia'].describe(percentiles=[0.25, 0.5, 0.75])
quartis_qtd_pessoas


# In[148]:


# Definindo os limites para cada faixa etária
faixas_qtd_pessoas = [1, 2, 3, 15]
rotulos_qtd_pessoas = ['1 pessoa', '2-3 pessoas', '3 ou mais']

# Adicionando uma nova coluna 'faixa_etaria' ao DataFrame
renda['faixas_qtd_pessoas'] = pd.cut(renda['qt_pessoas_residencia'], bins=faixas_qtd_pessoas, labels=rotulos_qtd_pessoas, right=False)


# In[153]:


#renda x quantidade de pessoas na residencia
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='faixas_qtd_pessoas', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita por Quantidade de Pessoas na Residência')
plt.xticks(rotation=45)
plt.legend(title='Qtd. Pessoas')
plt.show()


# O gráfico acima nos mostra que a renda rende a ser menor quanto mora apenas 1 pessoa na residência.

# In[ ]:





# In[172]:


#QUANTIDADE DE FILHOS
quartis_qtd_filhos = renda['qtd_filhos'].describe(percentiles=[0.25, 0.5, 0.75])
quartis_qtd_filhos


# Levando em consideração os percentis, acredito que podemos separar os grupos entre possui ou não filho.

# In[173]:


#criando a variável possui filho
renda['possui_filho'] = renda['qtd_filhos'].apply(lambda x: False if x == 0 else True)


# In[177]:


#renda x possui filhos
plt.figure(figsize=(12, 6))
sns.lineplot(x='data_ref', y='renda', hue='possui_filho', data=renda)
plt.xlabel('Data de Referência')
plt.ylabel('Renda')
plt.title('Renda predita por Possuir ou não Filhos')
plt.xticks(rotation=45)
plt.legend(title='Possui Filhos?')
plt.show()


# O gráfico acima, nosso ultimo gráfico, não parece possuir grande capacidade preditiva.

# In[ ]:





# ## Etapa 3 Crisp-DM: Preparação dos dados

# Inicialmente vamos ver informações gerais sobre os dados:

# In[105]:


renda.info()


# Claramente podemos perceber aqui que informações como o número de identificação do cliente, assim como o Unnamed não são variáveis importantes, logo, iremos descartá-las.

# In[106]:


renda=renda.drop('Unnamed: 0', axis=1)
renda=renda.drop('id_cliente', axis=1)
renda.head(1)


# Agora veremos os valores faltantes nas colunas que restaram:

# In[107]:


renda.isnull().sum()


# Podemos ver que a única coluna com dados faltantes é a coluna tempo de emprego, portanto, vamos analisar qual o tipo de renda desses clientes.

# In[108]:


#separando os clientes com tempo de emprego não preenchido:
tempo_emprego_nulo= renda[renda['tempo_emprego'].isnull()]

#buscando nesses, o tipo de renda:
tempo_emprego_nulo.groupby('tipo_renda').size()


# Podemos concluir que esses clientes não tem tempo de emprego preenchido por conta de serem pensionistas, portanto, podemos substituir esses valores faltantes por '0'.

# In[109]:


renda['tempo_emprego'].fillna(0, inplace=True)


# ## Etapa 4 Crisp-DM: Modelagem
# Nessa etapa que realizaremos a construção do modelo. Os passos típicos são:
# - Selecionar a técnica de modelagem
# - Desenho do teste
# - Avaliação do modelo
# 

# Pensando na etapa 'entendimento dos dados bivariada', os gráficos gerados com o streamlit já nos dão uma ideia de quais vaiáveis são relevantes para o nosso modelo, entretanto, vamos iniciar com uma regressão multipla para termos uma predição mais assertiva.
# 

# ### Regressão Multipla

# Para realizarmos a regressão multipla, precisamos incialmente analisar se nossas variáveis qualitativas tem como casela o seu valor mais frequente. Para os que não tem, utilizará-se um Treatment a fim de colocar a variável mais frequente como casela. 

# In[110]:


#1) vamos analisar qual valor mais recorrente na variável tipo_renda. 
renda.tipo_renda.value_counts()


# In[111]:


#agora vamos observar a matriz para ver se assalariado está como casela ou precisaremos mudar. 
y, x = patsy.dmatrices('np.log(renda) ~ C(tipo_renda)', data = renda)
x


# A variável analisada acima tem o seu valor mais frequente como casela, vamos para a próxima:

# In[112]:


#vamos analisar qual valor mais recorrente na variável educação
renda.educacao.value_counts()


# In[113]:


#agora vamos observar a matriz e tratá-la.
y, x = patsy.dmatrices('np.log(renda) ~ C(educacao, Treatment(2))', data = renda)
x


# In[114]:


#vamos analisar qual valor mais recorrente na variável tipo_residencia:
renda.tipo_residencia.value_counts()


# In[115]:


#agora vamos observar a matriz e tratá-la.
y, x = patsy.dmatrices('np.log(renda) ~ C(tipo_residencia, Treatment(1))', data = renda)
x


# In[116]:


#vamos analisar qual valor mais recorrente na variável estado_civil:
renda.estado_civil.value_counts()


# In[117]:


#agora vamos observar a matriz e tratá-la.
y, x = patsy.dmatrices('np.log(renda) ~ C(estado_civil, Treatment(0))', data = renda)
x


# Finalizamos o tratamento das nossas variáveis, agora vamos utilizar o modelo de regressão para decidir quais delas ficam. Lembrando que devemos nos atentar para o p-value, no qual são aceitos valores máximos de 5%.

# In[118]:


#1) regressão: renda explicado por todas as demais, incluindo as categorias e seus devidos tratamentos. 
reg1=smf.ols('renda~ sexo + posse_de_veiculo + posse_de_imovel + qtd_filhos + C(tipo_renda) + C(educacao, Treatment(2)) + C(estado_civil, Treatment(0)) + C(tipo_residencia, Treatment(1)) + idade + tempo_emprego + qt_pessoas_residencia', data = renda).fit()

reg1.summary() #resumo da regressão


# Analisando a regressão acima podemos perceber que algumas variáveis não tem forte correlação com renda, por isso, iremos manter abaixo apenas as variáveis com p-value máximo de 5%:

# In[119]:


#2) regressão: renda explicada pelas variáveis com p-value inferior a 5%
reg1=smf.ols('renda~ sexo + posse_de_imovel + idade + tempo_emprego', data = renda).fit()

reg1.summary() #resumo da regressão


# ## Etapa 5 Crisp-DM: Avaliação dos resultados
# 

# Com nossa última regressão, obtemos um R-quadrado menor que o anterior, entretanto, conseguimos eliminar diversas variáveis, tornando a nossa previsão mais genérica e adaptativa aos dados futuros. O nosso R-ajustado é igual ao R-quadrado, o que significa que ambos estão oferecendo uma avaliação equilibrada do desempenho do modelo, que explica 25,3% da variabilidade dos dados analisados.
# 

# ## Etapa 6 Crisp-DM: Implantação
# Nessa etapa colocamos em uso o modelo desenvolvido, normalmente implementando o modelo desenvolvido em um motor que toma as decisões com algum nível de automação.

# Agora que utilizamos a regressão para ter um direcionamento de quais variáveis utilizar no nosso modelo preditivo, vamos, a partir delas, treinar e executar uma árvore de regressão que possa nos mostrar quais caminhos serão utilizados para a predição final.

# ### Árvore
# 

# In[120]:


#criar um dataframe com as variáveis indicadas pelo modelo de regressão:
renda_modelo= renda[['posse_de_imovel', 'sexo', 'tempo_emprego', 'idade', 'renda']]


# In[121]:


#criar variáveis dummy para a variável 'sexo':
dummies_sexo = pd.get_dummies(renda_modelo['sexo'], prefix='sexo', drop_first=True)

#adicionar ao DataFrame original:
renda_modelo = pd.concat([renda_modelo, dummies_sexo], axis=1)

#remover a coluna original 'sexo':
renda_modelo.drop('sexo', axis=1, inplace=True)

print(renda_modelo.head())


# In[122]:


#separando x e y:
X= renda_modelo.drop(columns=['renda']) #todas exceto renda
y=pd.DataFrame(renda_modelo['renda']) 


# In[123]:


#separando em validação e teste:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2360873)


# In[124]:


#treinando duas árvores, uma com profundidade 2 e outra com profundidade 3:
# Fit regression model
#vamos rodar 2 arvores:
regr_1 = DecisionTreeRegressor(max_depth=2) #profundidade 2
regr_2 = DecisionTreeRegressor(max_depth=3) #profundidade 3

#treinar as duas com método fit:
regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)


# In[125]:


# mse1 = mean_squared_error(y_test, regr_1.predict(X_test))
#o método score retorna o coeficiente de determinação da árvore
mse1 = regr_1.score(X_train, y_train)
mse2 = regr_2.score(X_train, y_train)

template = "O R-quadrado da árvore com profundidade={0} é: {1:.2f}"

print(template.format(regr_1.get_depth(),mse1).replace(".",","))
print(template.format(regr_2.get_depth(),mse2).replace(".",","))


# In[126]:


#o método score retorna o coeficiente de determinação da árvore
mse1 = regr_1.score(X_test, y_test)
mse2 = regr_2.score(X_test, y_test)

template = "O R-quadrado da árvore com profundidade={0} é: {1:.2f}"

print(template.format(regr_1.get_depth(),mse1).replace(".",","))
print(template.format(regr_2.get_depth(),mse2).replace(".",","))


# Com base nas informações acima, percebemos que a árvore de profundidade 2 nos traz R-quadrados com valores mais próximos para treino e teste, assim como valores mais próximos da regressão múltipla realizada acima e que nos ajudou a escolher quais seriam as variáveis utilizadas para criar o dataframe atual. Entretanto, podemos plotar as duas para observação, visto que o R-quadrado da árvore com profundidade 3 ainda se mantém superior tanto na base de treino, quanto na de teste.

# In[127]:


#ÁRVORE DE PROFUNDIDADE 2
#utilizando a função plot tree
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('figure', figsize=(10, 10))
tp = tree.plot_tree(regr_1, #indica o nome da árvore
                    feature_names=X.columns,  #indica as colunas
                    filled=True) #opção estética


# In[128]:


#ÁRVORE DE PROFUNDIDADE 3
#utilizando a função plot tree
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('figure', figsize=(40, 40))
tp = tree.plot_tree(regr_2, #indica o nome da árvore
                    feature_names=X.columns,  #indica as colunas
                    filled=True) #opção estética


# Observando as árvores acima podemos perceber que a variável mais importante é tempo de emprego, visto que é nosso nó raiz. Posteriormente, como nó de decisão temos a variável sexo e, para a árvore de profundidade 2, finalmente o nó de termino que seria a previsão da renda. Já para a árvore de profundidade 3, temos a variável tempo de emprego aparecendo mais uma vez como nó de decisão, o que também é interessante, visto que por se tratar de renda temos várias possibilidades diferentes. 

# Julgando as árvores e seus R-quadrados, há de se considerar que a árvore de profundidade 3 pode ser mais eficiente, apesar de não ser tão genérica quanto a árvore de profundidade 2, ela também é uma árvore rasa que não apresenta tanto risco de overfitting. 
