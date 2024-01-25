#importando os pacotes necessários:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

#estilo dos gráficos:
sns.set(context='talk', style='ticks')

#confg iniciais da página, como título e ícone:
st.set_page_config(
     page_title="Análise exploratória",
     page_icon='https://as1.ftcdn.net/v2/jpg/00/37/32/38/1000_F_37323821_97gtccMK2eBOwM8yJ8mcLhDrl43am7Om.jpg', #copia endereço do link da imagem desejada
     layout="wide",
)

# Centralizar o título
st.markdown("<h1 style='text-align: center;'>Análise Exploratória</h1>", unsafe_allow_html=True)

#DataFrame:
renda = pd.read_csv('previsao_de_renda.csv')

#Texto:
st.write('## Gráfico de Linha Interativo:')


# Criando o filtro de INTERAÇÃO, aonde o usuário escolhe qual variável deseja visualizar:
hue_options = ['posse_de_imovel', 'posse_de_veiculo', 'qtd_filhos', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'sexo']
selected_hue = st.selectbox('Selecione uma variável para visualizar sua interação com a renda:', hue_options)

# Filtrar o DataFrame com base na escolha do usuário, para tornar esse processo mais seguro, faremos uma cópia do nosso DF:
filtered_renda = renda.copy()

# Configurar o gráfico
fig, ax = plt.subplots(figsize=(10, 7)) #tamanho da figura
sns.lineplot(x='data_ref', y='renda', hue=selected_hue, data=filtered_renda, ax=ax) #eixos
#ax.set_title(f'renda x {selected_hue}') #titulo
ax.set_xlabel('Data de Referência') 
ax.set_ylabel('Renda')
ax.tick_params(axis='x', rotation=45)
sns.despine()

# Mostrar o gráfico no Streamlit
st.pyplot(fig)

#--------------------------------------------------

st.write('## Gráficos Bivariada Interativo:')

# Permitir que o usuário escolha a variável para o eixo x
x_options = ['posse_de_imovel', 'posse_de_veiculo', 'qtd_filhos', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'sexo']
selected_x = st.selectbox('Selecione a variável que desejar para a interação com a variável renda:', x_options)

# Configurar o gráfico
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=selected_x, y='renda', data=renda, ax=ax)
#ax.set_title(f'Comparação da Variável Renda por {selected_x}')
ax.set_xlabel(selected_x)
ax.set_ylabel('Renda')
sns.despine()

# Mostrar o gráfico no Streamlit
st.pyplot(fig)
