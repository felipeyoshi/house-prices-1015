# Importar as bibliotecas necessárias
# Importar as bibliotecas necessárias
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px  # Biblioteca para criação de gráficos interativos

# Função para carregar o modelo treinado
def carregar_modelo():
    return joblib.load('modelo_treinado.pkl')

# Função para carregar o dataset
def carregar_dataset():
    return pd.read_csv('house_prices.csv')

# Função para criar gráficos de dispersão
def plotar_graficos(data):
    # Gráfico de dispersão para a feature 'Area'
    fig_area = px.scatter(data, x='Area', y='Price', title='Área vs Preço')
    st.plotly_chart(fig_area, use_container_width=True)
    
    # Boxplot para a feature 'Quartos'
    fig_quartos = px.box(data, x='Quartos', y='Price', title='Preço por Número de Quartos')
    st.plotly_chart(fig_quartos, use_container_width=True)
    
    # Boxplot para a feature 'Banheiros'
    fig_banheiros = px.box(data, x='Banheiros', y='Price', title='Preço por Número de Banheiros')
    st.plotly_chart(fig_banheiros, use_container_width=True)

# Função principal da aplicação
def main():
    # Título da aplicação
    st.title('Predição do Preço do Imóvel')
    
    # Subtítulo
    st.write('Este aplicativo prevê o preço do imóvel com base em sua área, número de quartos e banheiros.')
    
    # Carregar o dataset
    data = carregar_dataset()
    
    # Criar e mostrar os gráficos de dispersão
    plotar_graficos(data)

    # Entradas para as features
    area = st.number_input('Área do Imóvel (em m²)')  # Caixa de entrada para inserir a área do imóvel
    quartos = st.number_input('Número de Quartos')  # Caixa de entrada para inserir o número de quartos
    banheiros = st.number_input('Número de Banheiros')  # Caixa de entrada para inserir o número de banheiros
    
    # Botão para realizar a predição
    botao_predizer = st.button('Predizer Preço')
    
    # Quando o botão é pressionado
    if botao_predizer:
        # Carregar o modelo treinado
        modelo = carregar_modelo()
        
        # Criar um dataframe com as features inseridas
        features = pd.DataFrame([[area, quartos, banheiros]], columns=['Area', 'Quartos', 'Banheiros'])
        
        # Realizar a predição
        preco_predito = modelo.predict(features)
        
        # Mostrar o preço predito na tela
        st.subheader('Preço Predito do Imóvel:')
        st.write('R$ {:.2f}'.format(preco_predito[0]))

# Executar a aplicação
if __name__ == '__main__':
    main()
