# -*- coding: utf-8 -*-

# Análise Fatorial PCA

#%% Instalando os pacotes

# !pip install pandas
# !pip install numpy
# !pip install factor_analyzer
# !pip install sympy
# !pip install scipy
# !pip install matplotlib
# !pip install seaborn
# !pip install plotly
# !pip install pingouin
# !pip install pyshp

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
import sympy as sy
import scipy as sp


#%% Gráfico das cargas fatoriais (loading plot)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.05, point['y'], point['val'])
        
def plot_loading(df , var_list):
    """Plot loading factors.
    
    Parameters
    ----------
    df(DataFrame) : data to be ploted.
    var_list(list) : variable column name in df

    Returns
    -------
    (None)

    """
    plt.figure(figsize=(12,8))
    df_chart = df.reset_index()
    plt.scatter(df_chart[var_list[0]], df_chart[var_list[1]], s=50, color='red')



    label_point(x = df_chart[var_list[0]],
                y = df_chart[var_list[1]],
                val = df_chart['index'],
                ax = plt.gca()) 

    plt.axhline(y=0, color='grey', ls='--')
    plt.axvline(x=0, color='grey', ls='--')
    plt.ylim([-1.1,1.1])
    plt.xlim([-1.1,1.1])
    plt.title("Loading Plot", fontsize=16)
    plt.xlabel(var_list[0], fontsize=12)
    plt.ylabel(var_list[1], fontsize=12)
    plt.show()

#%% Importando o banco de dados

df = pd.read_excel("dados/notas_fatorial.xlsx")

#%% Informações sobre as variáveis

# Informações gerais sobre o DataFrame

print(df.info())

# Estatísticas descritiva das variáveis

print(df.describe())

#%% Separando somente as variáveis quantitativas do banco de dados

df_metricas = df[["finanças", "custos", "marketing", "atuária"]]

#%% Matriz de correlações de Pearson entre as variáveis

pg.rcorr(df_metricas, method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Outra maneira de analisar as informações das correlações

# Matriz de correlações em um objeto "simples"

corr = df_metricas.corr()

# Gráfico interativo

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text=corr.values,
        texttemplate='%{text:.4f}',
        colorscale='viridis'))

fig.update_layout(
    height = 600,
    width = 600,
    yaxis=dict(autorange="reversed"))

fig.show()

#%% Teste de Esfericidade de Bartlett
# Executa um teste de hipoteses.
# H0: matriz de correlação igual a identidade
# H1: sao diferentes

# Ver o p-valor em relação ao nível de significancia.

bartlett, p_value = calculate_bartlett_sphericity(df_metricas)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Definindo a PCA (procedimento inicial com todos os fatores possíveis)
#fatores maximos = nr de variaveis
fa = FactorAnalyzer(n_factors=4, method='principal', rotation=None).fit(df_metricas)

#%% Obtendo os eigenvalues (autovalores): resultantes da função FactorAnalyzer

autovalores = fa.get_eigenvalues()[0]

print(autovalores) # Temos 4 autovalores, pois são 4 variáveis ao todo

# Soma dos autovalores

round(autovalores.sum(), 2)


#%%

#%% Obtendo os autovalores e autovetores: ilustrando o fundamento

## Atenção: esta célula tem fins didáticos, não é requerida na FactorAnalyzer

# # Parametrizando o pacote

# lamda = sy.symbols('lamda')
# sy.init_printing(scale=0.8)

# # Especificando a matriz de correlações

# matriz = sy.Matrix(corr)
# polinomio = matriz.charpoly(lamda)

# polinomio

# # Obtendo as raízes do polinômio característico: são os autovalores

# autovalores, autovetores = sp.linalg.eigh(corr)
# autovalores_o = autovalores
# autovetores_o = autovetores

# autovalores = autovalores[::-1]

# # Obtendo os autovetores para cada autovalor extraído

# autovetores = autovetores[:, ::-1]

#%% Eigenvalues, variâncias e variâncias acumuladas

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Var Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Gráfico da variância acumulada dos componentes principais

plt.figure(figsize=(12,8))
ax = sns.barplot(x=tabela_eigen.index, y=tabela_eigen['Variância'], data=tabela_eigen, palette='rocket')
ax.bar_label(ax.containers[0])
plt.title("Fatores Extraídos", fontsize=16)
plt.xlabel(f"{tabela_eigen.shape[0]} fatores que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=12)
plt.ylabel("Porcentagem de variância explicada", fontsize=12)
plt.show()

#%% Determinando as cargas fatoriais. Representam a correlação de pearson entre as variáveis e cada variável

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = df_metricas.columns

print(tabela_cargas)

#%% Gráfico das cargas fatoriais (loading plot)

plot_loading(tabela_cargas, ['Fator 1' , 'Fator 2'])


#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = df_metricas.columns

print(tabela_comunalidades)

#%% Extração dos fatores para as observações do banco de dados

fatores = pd.DataFrame(fa.transform(df_metricas))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Adicionando os fatores ao banco de dados

df = pd.concat([df.reset_index(drop=True), fatores], axis=1)

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = df_metricas.columns

print(tabela_scores)

#%% Correlação entre os fatores extraídos

# A seguir, verifica-se que a correlação entre os fatores é zero (ortogonais)

pg.rcorr(df[['Fator 1','Fator 2', 'Fator 3', 'Fator 4']],
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Critério de Kaiser (raiz latente). Escolha da quantidade de fatores

# Verificar os autovalores com valores maiores que 1
# Existem dois componentes maiores do que 1

#%% Parametrizando a PCA para dois fatores (autovalores > 1)

fa = FactorAnalyzer(n_factors=2, method='principal', rotation=None).fit(df_metricas)

#%% Eigenvalues, variâncias e variâncias acumuladas de 2 fatores

# Note que não há alterações nos valores, apenas ocorre a seleção dos fatores

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Determinando as cargas fatoriais

# Note que não há alterações nas cargas fatoriais nos 2 fatores!

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = df_metricas.columns

print(tabela_cargas)

#%% Determinando as novas comunalidades

# As comunalidades são alteradas, pois há fatores retirados da análise!

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = df_metricas.columns

print(tabela_comunalidades)

#%% Extração dos fatores para as observações do banco de dados

# Vamos remover os fatores obtidos anteriormente

notas = df.drop(columns=['Fator 1', 'Fator 2', 'Fator 3', 'Fator 4'])

#  Vamos gerar novamente, agora para os 2 fatores extraídos

fatores = pd.DataFrame(fa.transform(df_metricas))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Adicionando os fatores ao banco de dados

notas = pd.concat([notas.reset_index(drop=True), fatores], axis=1)

# Note que são os mesmos, apenas ocorre a seleção dos 2 primeiros fatores!

#%% Identificando os scores fatoriais

# Não há mudanças nos scores fatoriais!

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = df_metricas.columns

print(tabela_scores)

#%% Criando um ranking (soma ponderada e ordenamento)

# O ranking irá considerar apenas os 2 fatores com autovalores > 1
# A base de seleção é a tabela_eigen

notas['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']

    notas['Ranking'] = notas['Ranking'] + notas[tabela_eigen.index[index]]*variancia
    
print(notas)

#%% Em certos casos, a "rotação de fatores" pode melhorar a interpretação

# Analisando pelo loading plot, aplica-se a rotação dos eixos na origem (0,0)
# O método mais comum é a 'varimax', que é a rotação ortogonal dos fatores
# O objetivo é aumentar a carga fatorial em um fator e diminuir em outro
# Em resumo, trata-se de uma redistribuição de cargas fatoriais

#%% Adicionando a rotação: rotation='varimax'

# Aplicando a rotação aos 2 fatores extraídos

fa_1 = FactorAnalyzer(n_factors=2, method='principal', rotation='varimax').fit(df_metricas)

cargas_fatoriais_1 = fa_1.loadings_

tabela_cargas_1 = pd.DataFrame(cargas_fatoriais_1)
tabela_cargas_1.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas_1.columns)]
tabela_cargas_1.index = df_metricas.columns

print(tabela_cargas_1)

#%% Gráfico das cargas fatoriais (loading plot)

plot_loading(tabela_cargas_1, ['Fator 1' , 'Fator 2'])



#%% Fim!