from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree

#%% validação cruzada e otimização dos hiperparametros

from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from sklearn.tree import DecisionTreeRegressor

def treinar_modelo(param):
  model = DecisionTreeRegressor(max_depth = param[0],
                                min_samples_split = param[1],
                                min_samples_leaf = param[2])
  
 
  score = cross_val_score(model, X, y, cv = 5)
                     
  score_validação = (np.mean(score))
  
  mse_validação = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
  
  # Convertendo para positivo (já que é a convenção negativa)
  mse_validação = -mse_validação
  
  mse_validação = (np.mean(mse_validação))
  
  # Imprimir a média dos MSEs
  print("Média do MSE durante a validação cruzada:", np.mean(mse_validação))
  
  r2_mean = np.mean(score)
  
  return -r2_mean

param = [(60, 100), #max_depth - profundidade máxima da arvore
         (6, 10), #min_samples_split - número minimo para dividr um ramo
         (8, 15)] #min_samples_leaf - número minimo de exemplos numa folha

opt = gp_minimize(treinar_modelo, 
                  param, 
                  random_state= 0, 
                  verbose = 1, 
                  n_calls = 50, 
                  n_random_starts = 15)

#%% criar o modelo final

import numpy as np

regressor = DecisionTreeRegressor(max_depth = opt.x[0],
                              min_samples_split = opt.x[1],
                              min_samples_leaf = opt.x[2])

regressor.fit(X, np.ravel(y))

scores = cross_val_score(regressor, 
                      X, 
                      y= np.ravel(y),
                      cv = 5) 

r2_mean = np.mean(scores)
mse_validação = cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_squared_error')
  # Convertendo para positivo (já que é a convenção negativa)
mse_validação = -mse_validação
  
mse_validação = (np.mean(mse_validação))
  
  # Imprimir a média dos MSEs
print("Média do MSE durante a validação cruzada:", np.mean(mse_validação))

print('r2 da validação cruzada:', r2_mean)

#%% predição nos dados de teste usando modelo final

y_pred = regressor.predict(X)

score_dados = regressor.score(X, y) #acurácia global
print (score_dados)

mse_dados = mean_squared_error(y, y_pred)
print('Mean Squared Error:', mse_dados)

#%% Montando uma lista com as métricas

métricas = {'score_validação': r2_mean, 'mse_validação': mse_validação, 'score_dados': score_dados, 'mse_dados': mse_dados}

#%%Plotar a árvore

# Nome das suas features
nome_das_features = ['Gênero', 'Idade', 'Grau de instrução', 'Gostar']

plt.figure(figsize=(25, 30))
plot_tree(regressor, feature_names=nome_das_features, filled=True, rounded=True, max_depth=65, fontsize=19)

# Salvar a figura em um arquivo PNG com alta resolução
plt.savefig('arvore_decisao.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Obtendo a importância das características

feature_importance = regressor.feature_importances_

# Exibindo a importância das características
for feature, importance in zip(X.columns, feature_importance):
    print(f'{feature}: {importance}')

#%% Gráfico para visualizar a importância das features
plt.figure(figsize=(10, 6))
bars = plt.barh(nome_das_features, feature_importance, color='skyblue')

# Aumentar o tamanho da fonte dos nomes das características
plt.yticks(fontsize=15)

# Definir os limites do eixo x de 0 a 1 (ou seja, de 0% a 100%)
plt.xlim(0, 1)

# Adicionar a porcentagem de importância de cada característica nas barras
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2%}', 
             va='center', ha='left', fontsize=15, color='black')

plt.xlabel('Importância das Características', fontsize=15)
plt.show()
