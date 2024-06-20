# -*- coding: utf-8 -*-

"""Provide general template to use in Clusterization.

@author: ulf Bergmann

"""

#%% Instalando os pacotes

# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install seaborn
# !pip install plotly
# !pip install scipy
# !pip install scikit-learn
# !pip install pingouin
# !python -m doctest template_clusterizacao.py

# !pip install seaborn --upgrade 
#%% Importando os pacotes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pingouin as pg
import plotly.express as px 
import plotly.io as pio

import sys
sys.path.append("./lib")
import funcoes_ulf as ulfpp

pio.renderers.default='browser'

pd.set_option('display.float_format', lambda x: '%.2f' % x)

#%% Importando o banco de dados
df_completo = pd.read_csv('dados/dados_paises.csv')


#%% Visualizando informações sobre a estrutura com os dados e variáveis 

print(df_completo.info())

# Estatísticas descritivas das variáveis.
df_completo.describe()

# Verificar a escala das variáveis para ver se é necessário normalizar / padronizar 
# Quando as variáveis estiverem em unidades de medidas ou escalas distintas

variables_list = list(df_completo.columns)


# Remover IDs e variáveis não numéricas
variables_list.remove('country')

ulfpp.plot_boxplot_for_variables(df_completo, variables_list)

normalize_variables = True

df_aux = df_completo.drop('country' , axis=1)

if normalize_variables:
    # Aplicando o procedimento de ZScore
    df_normalized = df_aux.apply(zscore, ddof=1)
    df_normalized.describe()
else:
    df_normalized = df_aux

#%% Gráfico 3D das observações

fig = px.scatter_3d(df_completo, 
                    x='child_mort', 
                    y='gdpp', 
                    z='total_fer',
                    text=df_completo.country)
fig.show()

# =============================================================================
# CLUSTERIZAÇÃO
# Opções para o método de encadeamento ("method"):
    ## single
    ## complete
    ## average

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

# =============================================================================



# Visualizando as distâncias. Apenas como informação
dist_euclidiana = pdist(df_normalized, metric='euclidean')

#%%

#%% Cluster hierárquico aglomerativo: distância euclidiana + single linkage
a_method = 'single'
a_metric = 'euclidean'

plt.figure(figsize=(16,8))
dend = sch.linkage(df_normalized, method = a_method , metric = a_metric )
dendrograma = sch.dendrogram(dend, color_threshold = 4.5, labels = list(df_completo.country))
plt.title('Dendrograma - ' + ' ' + a_method + ' / ' + a_metric , fontsize=16)
plt.xlabel('Paises', fontsize=16)
plt.ylabel('Distância', fontsize=16)

#apos analisar o grafico atualizar o valor da linha (escolha do nr de clusters)
plt.axhline(y = 4.5, color = 'red', linestyle = '--')
plt.show()

#%% Gerando a variável com a indicação do cluster no dataset
    
# Definir a quantidade de clusters analisando os dendrogramas
nr_clusters_escolhidos = 7
clusterizador = AgglomerativeClustering(n_clusters = nr_clusters_escolhidos, metric = a_metric, linkage = a_method)
indica_cluster = clusterizador.fit_predict(df_normalized)
df_completo['cluster - ' + a_method + ' - ' + a_metric] = indica_cluster
df_completo['cluster - ' + a_method + ' - ' + a_metric] = df_completo['cluster - ' + a_method + ' - ' + a_metric].astype('category')
df_normalized['cluster - ' + a_method + ' - ' + a_metric] = indica_cluster
df_normalized['cluster - ' + a_method + ' - ' + a_metric] = df_normalized['cluster - ' + a_method + ' - ' + a_metric].astype('category')

# Coeficientes do esquema hierárquico de aglomeração (single)
aux_coef = [y[1] for y in dendrograma['dcoord']]
print('Coeficintes do esquema hierárquico de aglomeração ' + a_method + ' - ' + a_metric)
print(aux_coef)

#Ulf
pontuacao_silhueta = silhouette_score(df_normalized, indica_cluster)
print(f"Pontuação da silhueta: {pontuacao_silhueta:.2f}")

#%% Cluster hierárquico aglomerativo: distância euclidiana + complete linkage

a_method = 'complete'
a_metric = 'euclidean'

plt.figure(figsize=(16,8))
dend = sch.linkage(df_normalized, method = a_method , metric = a_metric )
dendrograma = sch.dendrogram(dend, color_threshold = 4.5, labels = list(df_completo.country))
plt.title('Dendrograma - ' + ' ' + a_method + ' / ' + a_metric , fontsize=16)
plt.xlabel('Paises', fontsize=16)
plt.ylabel('Distância', fontsize=16)

#apos analisar o grafico atualizar o valor da linha (escolha do nr de clusters)
plt.axhline(y = 4.5, color = 'red', linestyle = '--')
plt.show()

#%% Gerando a variável com a indicação do cluster no dataset
    
# Definir a quantidade de clusters analisando os dendrogramas
nr_clusters_escolhidos = 7
clusterizador = AgglomerativeClustering(n_clusters = nr_clusters_escolhidos, metric = a_metric, linkage = a_method)
indica_cluster = clusterizador.fit_predict(df_normalized)
df_completo['cluster - ' + a_method + ' - ' + a_metric] = indica_cluster
df_completo['cluster - ' + a_method + ' - ' + a_metric] = df_completo['cluster - ' + a_method + ' - ' + a_metric].astype('category')
df_normalized['cluster - ' + a_method + ' - ' + a_metric] = indica_cluster
df_normalized['cluster - ' + a_method + ' - ' + a_metric] = df_normalized['cluster - ' + a_method + ' - ' + a_metric].astype('category')

# Coeficientes do esquema hierárquico de aglomeração (single)
aux_coef = [y[1] for y in dendrograma['dcoord']]
print('Coeficintes do esquema hierárquico de aglomeração ' + a_method + ' - ' + a_metric)
print(aux_coef)

#Ulf
pontuacao_silhueta = silhouette_score(df_normalized, indica_cluster)
print(f"Pontuação da silhueta: {pontuacao_silhueta:.2f}")

#%% Cluster hierárquico aglomerativo: distância euclidiana + average linkage

a_method = 'average'
a_metric = 'euclidean'

plt.figure(figsize=(16,8))
dend = sch.linkage(df_normalized, method = a_method , metric = a_metric )
dendrograma = sch.dendrogram(dend, color_threshold = 4.5, labels = list(df_completo.country))
plt.title('Dendrograma - ' + ' ' + a_method + ' / ' + a_metric , fontsize=16)
plt.xlabel('Paises', fontsize=16)
plt.ylabel('Distância', fontsize=16)

#apos analisar o grafico atualizar o valor da linha (escolha do nr de clusters)
plt.axhline(y = 4.5, color = 'red', linestyle = '--')
plt.show()

#%% Gerando a variável com a indicação do cluster no dataset
    
# Definir a quantidade de clusters analisando os dendrogramas
nr_clusters_escolhidos = 7
clusterizador = AgglomerativeClustering(n_clusters = nr_clusters_escolhidos, metric = a_metric, linkage = a_method)
indica_cluster = clusterizador.fit_predict(df_normalized)
df_completo['cluster - ' + a_method + ' - ' + a_metric] = indica_cluster
df_completo['cluster - ' + a_method + ' - ' + a_metric] = df_completo['cluster - ' + a_method + ' - ' + a_metric].astype('category')
df_normalized['cluster - ' + a_method + ' - ' + a_metric] = indica_cluster
df_normalized['cluster - ' + a_method + ' - ' + a_metric] = df_normalized['cluster - ' + a_method + ' - ' + a_metric].astype('category')

# Coeficientes do esquema hierárquico de aglomeração (single)
aux_coef = [y[1] for y in dendrograma['dcoord']]
print('Coeficintes do esquema hierárquico de aglomeração ' + a_method + ' - ' + a_metric)
print(aux_coef)

#Ulf
pontuacao_silhueta = silhouette_score(df_normalized, indica_cluster)
print(f"Pontuação da silhueta: {pontuacao_silhueta:.2f}")

#%%



#%% Cluster Não Hierárquico K-means

# Considerando que identificamos 3 possíveis clusters na análise hierárquica

kmeans = KMeans(n_clusters=7, init='random', random_state=100).fit(df_normalized)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans.labels_
df_completo['cluster_kmeans'] = kmeans_clusters
df_completo['cluster_kmeans'] = df_completo['cluster_kmeans'].astype('category')

#%% Identificando as coordenadas centroides dos clusters finais

cent_finais = pd.DataFrame(kmeans.cluster_centers_)
cent_finais.columns = df_normalized.columns
cent_finais.index.name = 'cluster'
cent_finais

#%% Plotando as observações e seus centroides dos clusters

plt.figure(figsize=(8,8))
sns.scatterplot(data=df_completo, x='gdpp', y='child_mort', hue='cluster_kmeans', palette='viridis', s=100)
sns.scatterplot(data=cent_finais, x='gdpp', y='child_mort', c = 'red', label = 'Centróides', marker="X", s = 40)
plt.title('Clusters e Centroides', fontsize=16)
plt.xlabel('gdpp', fontsize=16)
plt.ylabel('child_mort', fontsize=16)
plt.legend()
plt.show()

#%% Identificação da quantidade de clusters

# Método Elbow para identificação do nº de clusters
## Elaborado com base na "WCSS": distância de cada observação para o centroide de seu cluster
## Quanto mais próximos entre si e do centroide, menores as distâncias internas
## Normalmente, busca-se o "cotovelo", ou seja, o ponto onde a curva "dobra"

elbow = []
K = range(2,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(df_normalized)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(K)
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()


#%% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,11) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(df_normalized)
    silhueta.append(silhouette_score(df_normalized, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 11), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()


#%% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

# child_mort
pg.anova(dv='child_mort', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

# exports
pg.anova(dv='exports', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

# imports
pg.anova(dv='imports', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

# health
pg.anova(dv='health', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

# income
pg.anova(dv='income', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

# inflation
pg.anova(dv='inflation', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

# life_expec
pg.anova(dv='life_expec', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

# total_fer
pg.anova(dv='total_fer', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

# gdpp
pg.anova(dv='gdpp', 
         between='cluster - average - euclidean', 
         data=df_normalized,
         detailed=True).T

## A variável mais discriminante contém a maior estatística F (e significativa)
## O valor da estatística F é sensível ao tamanho da amostra
## A variável mais discriminante contém a maior estatística F (e significativa)
## O valor da estatística F é sensível ao tamanho da amostra

#%% Gráfico 3D dos clusters

# Perspectiva 1
fig = px.scatter_3d(df_completo, 
                    x='total_fer', 
                    y='income', 
                    z='life_expec',
                    color='cluster - average - euclidean')
fig.show()

# Perspectiva 2
fig = px.scatter_3d(df_completo, 
                    x='gdpp', 
                    y='income', 
                    z='life_expec',
                    color='cluster - average - euclidean')
fig.show()


#%% Nomear / definir caracteristicas principais dos clusters formados


# Agrupando o banco de dados
analise_paises = df_completo.drop(columns=['country'])
analise_paises = analise_paises.groupby(by=['cluster - average - euclidean'])

# Estatísticas descritivas por grupo

tab_desc_grupo = analise_paises.describe().T

#%% FIM