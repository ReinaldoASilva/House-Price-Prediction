import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Função para ler o arquivo CSV
def read_archive(kc_house_data):
    return pd.read_csv(kc_house_data)

# Função para analisar os dados
def analisar_dados(kc_house_data):
    data_house = read_archive(kc_house_data)
    
    # View information
    print("Resumo das informações gerais")
    print(data_house.info())

    # View descriptive statistics
    print("\nEstatísticas descritivas")
    print(data_house.describe())

    # View null values
    print("\nValores nulos por coluna")
    print(data_house.isnull().sum())

    return data_house  # Retorna o DataFrame após a análise

# Função para converter a coluna "date" em datetime
def converter_coluna_para_datetime(dataframe, coluna):
    dataframe[coluna] = pd.to_datetime(dataframe[coluna])
    return dataframe

# Função para excluir as colunas 'id', 'lat' e 'long'
def excluir_colunas(dataframe, colunas_a_excluir):
    dataframe = dataframe.drop(colunas_a_excluir, axis=1)
    return dataframe

# Nome do arquivo CSV
kc_house_data = "kc_house_data.csv"

# Chama a função para analisar os dados
kc_house_data = analisar_dados(kc_house_data)

# Chama a função para converter a coluna "date" em datetime
kc_house_data = converter_coluna_para_datetime(kc_house_data, "date")

# Lista das colunas a serem excluídas
colunas_a_excluir = ['id', 'lat', 'long']

# Chama a função para excluir as colunas especificadas
kc_house_data = excluir_colunas(kc_house_data, colunas_a_excluir)

# Exibe o DataFrame após as operações
print("\nDataFrame Após as Operações:")
print(kc_house_data)

#------------------------------------ Resumo das estatisticas descritivas------------------------

# Histograma do preço da casa
plt.figure(figsize=(8,6))
plt.hist(kc_house_data["price"], bins=30, color="blue")
plt.title("Distribuição do preço")
plt.xlabel("Preço (USD)")
plt.ylabel("Contagem")
plt.show()

# Cálculo das correlações
correlation_matrix = kc_house_data.corr()

# Identificar as variáveis mais correlacionadas com o preço
price_correlation = correlation_matrix['price'].sort_values(ascending=False)

# Selecionar as variáveis independentes mais correlacionadas
top_features = price_correlation[1:6]  # Pode ajustar o número de variáveis

# Visualizar as variáveis selecionadas
print(top_features)

# Criar um novo DataFrame com as variáveis selecionadas
selected_features = kc_house_data[top_features.index]


# Heatmap de correlações
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlações entre variáveis")
plt.show()


# Gráfico de barras para a variável "waterfront"
# Nesse gráfico podemos ver que temos poucas casas de frente para água, o que revela um grande desequilibrio das classes
plt.figure(figsize=(12,10))
sns.countplot(data=kc_house_data, x="waterfront")
plt.title("Distribuição da variável waterfront")
plt.xlabel("Vista para a água")
plt.ylabel("Contagem")
plt.show()


# Box plot dos preços com e sem vista para a água
# Nesse caso podemos ver que as casas de vista para água tendem a ser mais caras
plt.figure(figsize=(8, 6))
sns.boxplot(data=kc_house_data, x='waterfront', y='price')
plt.title('Preços de Casas com e sem Vista para a Água')
plt.xlabel('Vista para a Água')
plt.ylabel('Preço')
plt.show()

# Estatísticas descritivas separadas para casas com e sem vista para a água
with_waterfront = kc_house_data[kc_house_data['waterfront'] == 1]
without_waterfront = kc_house_data[kc_house_data['waterfront'] == 0]

print("Estatísticas para Casas com Vista para a Água:")
print(with_waterfront.describe())

print("\nEstatísticas para Casas sem Vista para a Água:")
print(without_waterfront.describe())


#---------------------------------------- Aplicar a transformação logarítmica------------------------------

# Como os valores estavam muito descrepante achei interessante fazer uma transort
kc_house_data['log_price'] = np.log(kc_house_data['price'])

# Plotar um histograma do preço original e do preço transformado
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(kc_house_data['price'], bins=30, color='blue', alpha=0.7)
plt.title('Preço Original')
plt.xlabel('Preço')
plt.ylabel('Contagem')

plt.subplot(1, 2, 2)
plt.hist(kc_house_data['log_price'], bins=30, color='green', alpha=0.7)
plt.title('Preço Transformado (Logaritmo)')
plt.xlabel('Log(Preço)')
plt.ylabel('Contagem')

plt.tight_layout()
plt.show()

#--------------------------------------------Criação do modelo----------------------------------------

# Separar variáveis independentes (X) e a variável de destino (y)
X = kc_house_data[['sqft_living', 'grade', 'sqft_above','sqft_living15']]  # Exemplo de variáveis independentes
y = kc_house_data['log_price']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R²: {r2}')


# Treinar um modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # X_train contém as variáveis selecionadas anteriormente

# Calcular a importância das variáveis
importances = model.feature_importances_

# Criar um DataFrame para visualização
importance_df = pd.DataFrame({'Variável': X_train.columns, 'Importância': importances})

# Ordenar o DataFrame por importância
importance_df = importance_df.sort_values(by='Importância', ascending=False)

# Visualizar a importância das variáveis em um gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='Importância', y='Variável', data=importance_df)
plt.title('Importância das Variáveis no Modelo de Random Forest')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.show()
'''
grade: Importância de aproximadamente 40%.
sqft_living: Importância de aproximadamente 25%.
sqft_above: Importância de aproximadamente 20%.
sqft_living15: Importância de aproximadamente 15%.
'''
