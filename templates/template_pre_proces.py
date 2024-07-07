# -*- coding: utf-8 -*-

"""Provide a general template to use in ML data preparation.

##Processo Padrão:
    
###Informações iniciais - Estatística Descritiva.
    
###Ajuste Dos Tipos Das Variáveis.
    
###Tratamento De Valores Ausentes.
    
###Tratamento Das Variáveis Métricas.

###Tratamento Das Variáveis Categóricas.

###Verificar Se As Escalas Dos Valores São Similares E Padronizar Os Valores.

###Eliminação De Variáveis / Redução De Dimensão.

###Verificar o Vazamento De Informação (Data Leakage).

###Eliminação De Variáveis Colineares (Alta Correlação).

###Verificar possibilidade de utilizar Análise De Componentes Principais (PCA).

@author: Ulf Bergmann

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

sys.path.append("./templates/lib")
print(sys.path)

import funcoes_ulf as ulfpp


np.random.seed(42)  # semente de aleatoriedade

# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', 20)

#%%

# =============================================================================
# Leitura dos dados
# =============================================================================

# o conjunto de dados está no Google Drive, em um CSV separado por ,
df = pd.read_csv('cases/titanic/dados/train.csv', sep = ',')


#%%
# =============================================================================
# Informações iniciais - Estatística Descritiva
# =============================================================================


print("************* ANÁLISE INICIAL ***********************\n" , )
print("---- Exemplo dos dados -----")
print(df.head(10))

print("\n---- Informações das variáveis -----")
print(df.info())

print("\n---- Estatísticas Descritivas -----")
print(df.describe())

# =============================================================================
# AJUSTE DOS TIPOS DAS VARIAVEIS
# =============================================================================

# banco['VAR'] = pd.to_datetime(banco['VAR_STR'])
# banco['VAR'] = pd.to_numeric(banco['VAR_STR'], errors='coerce')

# =============================================================================
# TRATAMENTO DE VALORES AUSENTES
# =============================================================================

print("\n---- Valores null originais -----")
print(df.isnull().sum())

#preencher com o valor que mais aparece
df['Embarked'].fillna((df['Embarked'].mode()[0]), inplace = True)

#preencher com a media dos valores. Somente variaveis metricas
#df['A'].fillna((df['A'].mean()), inplace = True)

#extrair de informações de outras variáveis, p.ex., da média de cada honorifico em name
def extract_honorific(name):
    record = False
    honorific = ''
    for i, char in enumerate(name):
        if char == ',':
            record = True
        if char == '.':
            record = False
        if record == True:
            honorific += name[i + 2]
    return honorific[:-1]

def gerar_honorific_e_age_com_media(dados):
    
    #Finding the honorifics of all the passengers:
    honorifics = [extract_honorific(name) for name in dados.Name]

    #Creating a new "Honorific" column:
    dados.insert(3 , "Honorific", honorifics)
    
    #COmpletar a idade com a média da idade de acordo com o honorifico que esta no meio do nome.

    median_ages = pd.Series(dados.groupby(by = 'Honorific')['Age'].median())
    median_ages.sort_values(ascending = False)
    
    for i, row in dados.iterrows():
        
        anAge = row['Age']
        anHonorific = row['Honorific']
        if  pd.isnull(anAge):
            if anHonorific in median_ages:  
                dados.at[i , 'Age'] = median_ages[anHonorific]
    
gerar_honorific_e_age_com_media(df)

print("\n---- Valores null finais -----")
print(df.isnull().sum())


# =============================================================================
# TRATAMENTO DAS VARIÁVEIS MÉTRICAS
# =============================================================================

df.info()

lista_variáveis_metricas = ['Age','Fare']
# lista_variáveis_metricas = list(banco.columns)


# =============================================================================
# TRATAMENTO DAS VARIÁVEIS CATEGÓRICAS
# =============================================================================


# Identificação das variáveis categoricas e metricas 
a = ulfpp.search_for_categorical_variables(df)
print(a)
lista_variaveis_categoricas = ['Survived' , 'Pclass' , 'Sex' ,'SibSp' , 'Parch','Cabin', 'Embarked' ]

# preencher categorias com um codigo inteiro para cada uma
for i , j in enumerate(lista_variaveis_categoricas):
    df[j] = ulfpp.fill_categoric_field_with_value(df[j] , True)



# =============================================================================
# # Verificar se as escalas dos valores são similares, ou seja, se não tem uma 
#  que vá de -10 a 50 e outra de 0 a 100000. 
#
# Se forem discrepantes devemos padronizar os dados
# MinMaxScaler => coloca os valores entre 0 e 1
# StandardScaler (z-score ) => coloca os valores com média 0 e desvio  padrao 1
# FunctionTransformer(np.log1p) => aplica a funcao Log(x) nos valores 
# 
# Min-max normalization is preferred when data doesn’t follow Gaussian or normal distribution. It’s favored for normalizing algorithms that don’t follow any distribution, such as KNN and neural networks. Note that normalization is affected by outliers.
# Standardization can be helpful in cases where data follows a Gaussian distribution. However, this doesn’t necessarily have to be true. In addition, unlike normalization, standardization doesn’t have a bounding range. This means that even if there are outliers in data, they won’t be affected by standardization.
# Log scaling is preferable if a dataset holds huge outliers.
# =============================================================================

#verificar as escalas das variaveis metricas
ulfpp.plot_boxplot_for_variables(df , [lista_variáveis_metricas])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(df)

sc.mean_

sc.var_

df_normalizado = pd.DataFrame(sc.transform(df) , columns=df.columns)

df_normalizado.describe()


# =============================================================================
# ELIMINAÇÃO DE VARIÁVEIS / REDUÇÃO DE DIMENSÃO
#
# Preditores plausíveis:# Pré selecionar variáveis que sejam preditoras plausíveis
#   (bom senso do pesquisador).
# Coincidências acontecem em análises de big data e pode ser que o algoritmo 
#   dê muita importância para associações espúrias.
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
df.drop(['PassengerId'] , inplace=True , axis=1)

# =============================================================================
# ELIMINAÇÃO DE VARIÁVEIS COLINEARES (ALTA CORRELAÇÃO)
# Variáveis colineares trazem informação redundante. Além disso, aumentam a instabilidade dos modelos.
# Estabelecer um limite de correlação com alguma outra variável (0,75 a 0,90).
# =============================================================================


lista_maiores_correlacoes_continuas , corr_matrix = ulfpp.analyse_correlation_continuos_variables(df , lista_variáveis_metricas , 10)

ulfpp.plot_correlation_heatmap(df , lista_variáveis_metricas)
print("\n Correlaçoes das Variaveis Continuas")
print(lista_maiores_correlacoes_continuas)

#%%
df_p_value , maiores_correlacoes_categoricas = ulfpp.analyse_plot_correlation_categorical_variables (df , lista_variaveis_categoricas)
print("\n Correlaçoes das Variaveis Categoricas")
print(maiores_correlacoes_categoricas)


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
df_filtrado = df[df['PopResid'] > 10000]

ulfpp.plot_frequencias_valores_atributos(df_filtrado , ['PopResid'] )
df_filtrado.describe().T


