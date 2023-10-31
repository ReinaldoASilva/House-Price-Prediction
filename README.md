![alt text](image.jpeg)

# Análise de Dados e Modelagem de Preços de Imóveis

* Este projeto tem como objetivo realizar uma análise de dados e construir um modelo de regressão linear para prever os preços de imóveis com base em diversas variáveis. A análise e modelagem foram conduzidas utilizando a linguagem de programação Python e as bibliotecas, Pandas, NumPy, Matplotlib e Seaborn, além do uso do scikit-learn para construir o modelo de regressão.

### Dataset

#### O dataframe usado contém as seguintes colunas:

* id: Identificador único para cada casa.
date: Data da venda da casa.
price: Preço de venda da casa.
bedrooms: Número de quartos.
bathrooms: Número de banheiros.
sqft_living: Área interna da casa em pés quadrados.
sqft_lot: Área do terreno em pés quadrados.
floors: Número de andares na casa.
waterfront: Indicador se a casa tem vista para a água.
view: Classificação da vista da casa.
condition: Classificação da condição da casa.
grade: Classificação da qualidade da construção e design da casa.
sqft_above: Área acima do solo em pés quadrados.
sqft_basement: Área do porão em pés quadrados.
yr_built: Ano de construção da casa.
yr_renovated: Ano de renovação da casa.
zipcode: Código postal da localização da casa.
lat: Latitude da localização da casa.
long: Longitude da localização da casa.
sqft_living15: Área interna média das 15 casas mais próximas em pés quadrados.
sqft_lot15: Área média do terreno das 15 casas mais próximas em pés quadrados.
Análise Exploratória de Dados

Depois de verificar as condições do dataframe em relação a valores nulos, duplicados, alteração da coluna date e exclusão de algumas colunas realize uma análise exploratória para entender a distribuição, tendências e relações entre as variáveis. Segue o resultado dessa análise:

As variáveis independentes mais correlacionadas com o preço das casas, incluindo:
grade: Importância de aproximadamente 40%.
sqft_living: Importância de aproximadamente 25%.
sqft_above: Importância de aproximadamente 20%.
sqft_living15: Importância de aproximadamente 15%.
Modelagem

Construímos um modelo de regressão linear usando as variáveis independentes. 

Segue os Resultados do modelo de regressão linear utilizando as variáveis independentes:
RMSE (Root Mean Square Error): 0.3467
R² (Coeficiente de Determinação): 0.5783
Insights

Com base nos resultados, podemos tirar os seguintes insights:

A qualidade da classificação da casa (grade) é a variável mais influente nos preços das casas, representando cerca de 40% da importância.
A metragem quadrada da sala de estar (sqft_living) é a segunda variável mais importante, com aproximadamente 25% de importância.
A área acima do solo (sqft_above) também desempenha um papel significativo, com cerca de 20% de importância.
A metragem quadrada da sala de estar de vizinhos próximos (sqft_living15) tem uma influência menor, mas ainda é relevante, com aproximadamente 15% de importância.
Próximos Passos

Considerar a exploração de modelos mais avançados, como regressão polinomial, modelos de séries temporais ou modelos de aprendizado profundo.
Realizar mais engenharia de recursos para melhorar o desempenho do modelo.
Continuar refinando o modelo e realizando análises adicionais para aprimorar a capacidade de previsão.
