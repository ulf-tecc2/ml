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
from IPython.display import display


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

import funcoes_ulf as ulfpp


np.random.seed(42)  # semente de aleatoriedade

# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', 20)

#%%

# =============================================================================
# Leitura dos dados
# =============================================================================

# o conjunto de dados está no Google Drive, em um CSV separado por ;
df = pd.read_csv('dados/ofertas_relampago.csv', sep = ',')

#%%
# =============================================================================
# Informações iniciais 
# =============================================================================
print("************* ANÁLISE INICIAL ***********************\n" , )
print("---- Exemplo dos dados -----")
display(df.head(5))

print("\n---- Informações das variáveis -----")
print(df.info())

print("\n---- Estatísticas Descritivas -----")
print(df.describe())



#%% AJUSTE DOS TIPOS DAS VARIAVEIS

df['OFFER_START_DATE'] = pd.to_datetime(df['OFFER_START_DATE'])
df['OFFER_START_DTTM'] = pd.to_datetime(df['OFFER_START_DTTM'])
df['OFFER_FINISH_DTTM'] = pd.to_datetime(df['OFFER_FINISH_DTTM'])

#verificar se a OFFER_START_DATE sempre é a mesma que a data em OFFER_START_DTTM
df['dif'] = df.apply(lambda row: row['OFFER_START_DTTM'].date() == row['OFFER_START_DATE'].date() , axis=1)
df['dif'].unique()
# todas as datas são iguais, podemos retirar a coluna OFFER_START_DATE_date
df.drop(['OFFER_START_DATE' , 'dif'] , inplace=True , axis=1)

df["SOLD_AMOUNT"] = df["SOLD_AMOUNT"].fillna(0)

#%% AJUSTE DOS VALORES NAN E NONE

print("\n---- Valores null -----")
print(df.isnull().sum())


# =============================================================================
# INVOLVED_STOCK - Estoque inicial
# REMAINING_STOCK_AFTER_END - Estoque depois da campanha
# SOLD_QUANTITY - Quantidade Vendida - nan significa 0 => trocar nan
# SOLD_AMOUNT - Valor Vendido - nan significa 0 => trocar nan
# =============================================================================
df['SOLD_QUANTITY'] = df['SOLD_QUANTITY'].fillna(0)
df['SOLD_AMOUNT'] = df['SOLD_AMOUNT'].fillna(0)


#%%   ELIMINAÇÃO DE VARIÁVEIS / REDUÇÃO DE DIMENSÃO
# =============================================================================
# Preditores plausíveis:# Pré selecionar variáveis que sejam preditoras plausíveis (bom senso do pesquisador).
# Coincidências acontecem em análises de big data e pode ser que o algoritmo dê muita importância para associações espúrias.
# =============================================================================

#Variavel OFFER_TYPE tem o mesmo valor para todos os registros. Remover pois
#não acrescenta informação nenhuma
df.drop(['OFFER_TYPE'] , inplace=True , axis=1)


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


# Identificação das variáveis categoricas e metricas 
a = ulfpp.search_for_categorical_variables(df)
print(a)

lista_variaveis_categoricas = ['ORIGIN' , 'SHIPPING_PAYMENT_TYPE' , 'DOM_DOMAIN_AGG1' ,'VERTICAL' , 'DOMAIN_ID' ]
lista_variáveis_metricas = ['INVOLVED_STOCK','REMAINING_STOCK_AFTER_END','SOLD_AMOUNT','SOLD_QUANTITY']

# =============================================================================
# ELIMINAÇÃO DE VARIÁVEIS COLINEARES (ALTA CORRELAÇÃO)
# Variáveis colineares trazem informação redundante. Além disso, aumentam a instabilidade dos modelos.
# Estabelecer um limite de correlação com alguma outra variável (0,75 a 0,90).
# =============================================================================

#%%

lista_maiores_correlacoes_continuas , corr_matrix = ulfpp.analyse_correlation_continuos_variables(df , lista_variáveis_metricas , 10)

ulfpp.plot_correlation_heatmap(df , lista_variáveis_metricas)

print("\n Correlaçoes das Variaveis Continuas")
print(lista_maiores_correlacoes_continuas)

# INVOLVED_STOCK e REMAINING_STOCK_AFTER_END tem correlação de 0,99
df.drop(['REMAINING_STOCK_AFTER_END'] , inplace=True , axis=1)

#%%
ulfpp.plot_frequencias_valores_atributos(df , lista_variaveis_categoricas )
#%%
ulfpp.print_count_cat_var_values(df, lista_variaveis_categoricas)

#%%preencher categorias com um codigo inteiro para cada uma
df['ORIGIN'] = ulfpp.fill_categoric_field_with_value(df['ORIGIN'] , True)
df['SHIPPING_PAYMENT_TYPE'] = ulfpp.fill_categoric_field_with_value(df['SHIPPING_PAYMENT_TYPE'] , True)
df['DOM_DOMAIN_AGG1'] = ulfpp.fill_categoric_field_with_value(df['DOM_DOMAIN_AGG1'] , True)
df['VERTICAL'] = ulfpp.fill_categoric_field_with_value(df['VERTICAL'] , True)
df['DOMAIN_ID'] = ulfpp.fill_categoric_field_with_value(df['DOMAIN_ID'] , True)

#%%
lista_variaveis_categoricas = ['ORIGIN' , 'SHIPPING_PAYMENT_TYPE' , 'DOM_DOMAIN_AGG1' ,'VERTICAL'  ]

lista_maiores_correlacoes_categoricas = ulfpp.analyse_plot_correlation_categorical_variables (df , lista_variaveis_categoricas)
print("\n Correlaçoes das Variaveis Categoricas")
print(lista_maiores_correlacoes_categoricas)
#%%


# =============================================================================
# Análise de Componentes Principais. Técnica de aprendizado não supervisionado. 
# O objetivo é encontrar combinações lineares das variáveis preditoras que incluam a maior quantidade
# possível da variância original. Cria componentes principais não correlacionados.
# 
# Ver template_pca.py
# =============================================================================







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
