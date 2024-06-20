# -*- coding: utf-8 -*-

"""Provide a general template to use in ML data preparation.

@author: ulf Bergmann

"""


#%%
# =============================================================================
# INICIALIZAÇÃO DO AMBIENTE
# =============================================================================
import pandas as pd # para processamento de bancos de dados
import numpy as np # para processamento numérico de bancos de dados
import matplotlib.pyplot as plt # para geração de gráficos
from matplotlib import rc  # configurações adicionais para os gráficos a serem gerados



# importamos a funcionalidade de split do conjunto de dados em treino/teste
from sklearn.model_selection import train_test_split

# definimos o estilo dos gráficos
# mais estilos em https://matplotlib.org/3.1.1/gallery/#style-sheets
plt.style.use("fivethirtyeight")
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10}) #fonte utilizada
rc('mathtext',**{'default':'regular'})

import warnings   # ignorando os warnings emitidos pelo Python
warnings.filterwarnings("ignore")

import sys

sys.path.append("./lib")
print(sys.path)

import funcoes_ulf as ulfpp


np.random.seed(42)  # semente de aleatoriedade

# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#%%
'''
Leitura dos dados
'''
# =============================================================================
# Leitura dos dados
# =============================================================================

# o conjunto de dados está no Google Drive, em um CSV separado por ;
banco = pd.read_csv('https://drive.google.com/uc?export=download&id=1y6ESHpXVMd15a5aiJ4hVs412vmZmk_EV', sep = ';')
#banco = pd.read_csv('C:/Users/ulf/OneDrive/Python/titanic/dados/train.csv')
banco = pd.read_csv("https://drive.google.com/uc?export=download&id=1kErGPTcfCJotNKceFIDW0Q_qLgNJlGM8")
#%%
# =============================================================================
# Informações iniciais 
# =============================================================================
print("************* ANÁLISE INICIAL ***********************\n" , )
print("---- Exemplo dos dados -----")
print(banco.head(10))

print("\n---- Informações das variáveis -----")
print(banco.info())

print("\n---- Estatísticas Descritivas -----")
print(banco.describe())

print("\n---- Valores null -----")
print(banco.isnull().sum())


#%%   ELIMINAÇÃO DE VARIÁVEIS / REDUÇÃO DE DIMENSÃO
# =============================================================================
# Preditores plausíveis:# Pré selecionar variáveis que sejam preditoras plausíveis (bom senso do pesquisador).
# Coincidências acontecem em análises de big data e pode ser que o algoritmo dê muita importância para associações espúrias.
# =============================================================================


# =============================================================================
# Cuidado com vazamento de informação (data leakage)
#  Acontece quando os dados de treino apresentam informação escondida que faz com que o modelo
# aprenda padrões que não são do seu interesse.
# Uma variável preditora tem escondida o resultado certo:
# Não é a variável que está predizendo o desfecho, mas o desfecho que está predizendo
# ela.
# Exemplo: identificadores (id)
     # Incluir o número identificador do paciente como variável preditora
     # Problema: Se pacientes de hospital especializado em câncer tiverem números semelhantes.
     #           Se o objetivo for predizer câncer, algoritmo irá dar maior probabilidade a esses pacientes.
# =============================================================================

#variavel que eh id
banco.drop(['PassengerId'] , inplace=True , axis=1)

# Identificação das variáveis categoricas e metricas 
a = ulfpp.search_for_categorical_variables(banco)
print(a)
lista_variaveis_categoricas = ['Survived' , 'Pclass' , 'Sex' ,'SibSp' , 'Parch','Embarked' ]

banco.info()

lista_variáveis_metricas = ['Age','Fare']

# =============================================================================
# ELIMINAÇÃO DE VARIÁVEIS COLINEARES (ALTA CORRELAÇÃO)
# Variáveis colineares trazem informação redundante. Além disso, aumentam a instabilidade dos modelos.
# Estabelecer um limite de correlação com alguma outra variável (0,75 a 0,90).
# =============================================================================

#%%
lista_variáveis_metricas = list(banco.columns)
lista_variáveis_metricas.remove('VERDATE')

lista_maiores_correlacoes_continuas , corr_matrix = ulfpp.analyse_correlation_continuos_variables(banco , lista_variáveis_metricas , 10)

ulfpp.plot_correlation_heatmap(banco , lista_variáveis_metricas)

print("\n Correlaçoes das Variaveis Continuas")
print(lista_maiores_correlacoes_continuas)

lista_maiores_correlacoes_categoricas = ulfpp.analyse_plot_correlation_categorical_variables (banco , lista_variaveis_categoricas)
print("\n Correlaçoes das Variaveis Categoricas")
print(lista_maiores_correlacoes_continuas)
#%%



# Análise de Componentes Principais. Técnica de aprendizado não supervisionado. 
# O objetivo é encontrar ombinações lineares das variáveis preditoras que incluam a maior quantidade
# possível da variância original. Cria componentes principais não correlacionados.







#%% 
# =============================================================================
# Reducao dos dados. Eliminar linhas que contenham pouca quantidade de informação 
# Exemplo, munícipios com baixa população e que provavelmente as amostras utilizadas sejam pequenas
# 
# =============================================================================
banco_filtrado = banco[banco['PopResid'] > 10000]
ulfpp.plot_frequencias_valores_atributos(banco_filtrado , ['PopResid'] )
banco_filtrado.describe().T


#%% Separar dados de treino e teste

#variavel a ser predita
outcome = banco_filtrado['ExpecVida']

#Eliminar variáveis que
banco_filtrado.drop(['ExpecVida' , 'cod_municipio'] , axis=1 , inplace=True)

# fazemos a separação do conjunto de dados em treino/teste, com 30% dos dados para teste
X_train, X_test, y_train, y_test = train_test_split(banco_filtrado, outcome, test_size=0.3)

#%%
# =============================================================================
# # Verificar se as escalas dos valores são similares, ou seja, se não tem uma que vá de -10 a 50 e outra de 0 a 100000. 
#
# # Se forem discrepantes devemos padronizar os dados
# # MinMaxScaler => coloca os valores entre 0 e 1
# # StandardScaler (z-score ) => coloca os valores com média 0 e desvio  padrao 1
# # FunctionTransformer(np.log1p) => aplica a funcao Log(x) nos valores 
# 
# Min-max normalization is preferred when data doesn’t follow Gaussian or normal distribution. It’s favored for normalizing algorithms that don’t follow any distribution, such as KNN and neural networks. Note that normalization is affected by outliers.
# 
# Standardization can be helpful in cases where data follows a Gaussian distribution. However, this doesn’t necessarily have to be true. In addition, unlike normalization, standardization doesn’t have a bounding range. This means that even if there are outliers in data, they won’t be affected by standardization.
# 
# Log scaling is preferable if a dataset holds huge outliers.
# =============================================================================

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

sc.mean_

sc.var_

X_train = pd.DataFrame(sc.transform(X_train) , columns=X_train.columns)

X_train.describe()

X_test = pd.DataFrame(sc.transform(X_test) , columns=X_test.columns)
X_test.describe()

a = ulfpp.search_for_categorical_variables(banco)

print(a)
