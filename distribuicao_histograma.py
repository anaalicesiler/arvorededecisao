import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_excel('dados.xlsx')

#%% Selecionar colunas específicas para o resumo estatístico das variavéis independentes númericas

colunas_selecionadas = ['Idade', 'Gostar']
resumo_estatistico = df[colunas_selecionadas].describe()

#%% Montar a matriz X e y

X = df.copy()
X = X.drop(['Consumidores', Nota_de_aceitacao], axis = 1)       #deixando apenas as variáveis independentes

y = df[['Nota_de_aceitacao']] #variavél dependente

#%% Histogramas das variáveis independentes

X.hist(bins=20, figsize=(15, 10))
plt.show()

#%% # Agrupar em aceitção, neutralidade, e rejeição

# Definir as faixas de categorias
faixas = [0, 2, 3, 5]  # 0 a 2 (rejeição), 3 (indiferença), 4 a 5 (aceitação)

# Criar rótulos para as categorias
rotulos = ['Rejeição', 'Indiferença', 'Aceitação']

# Adicionar uma nova coluna 'Categoria' ao DataFrame
df['Categoria'] = pd.cut(df['Hedônica Tradicional'], bins=faixas, labels=rotulos, include_lowest=True)

# Contar a quantidade de observações em cada categoria
contagem_por_categoria = df['Categoria'].value_counts()

#Gráfico

# Calcular porcentagens
porcentagens = contagem_por_categoria / contagem_por_categoria.sum() * 100

# Criar o gráfico
plt.figure(figsize=(4, 2))
bars = porcentagens.plot(kind='bar', color='skyblue')
plt.ylabel('Porcentagem')

# Adicionar porcentagem em cima de cada barra
for bar in bars.patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom')

plt.xticks(rotation=0)
plt.ylim(0, 100)  # Ajusta o intervalo do eixo y para 0-100%
plt.tight_layout()

# Mostrar o gráfico
plt.show()
