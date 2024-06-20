# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:04:06 2022

@author: bergmann
"""


import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

#ler e combinar os arquivos
df_treino  = pd.read_csv('C:/Users/bergmann/OneDrive/Python/spaceship/train.csv')

df_teste  = pd.read_csv('C:/Users/bergmann/OneDrive/Python/spaceship/test.csv')

y_treino = df_treino.Transported.copy()

df_combinado = pd.concat( [df_treino , df_teste] , axis=0 , ignore_index=True)

#%% gerar um id de cada grupo que viaja junto
df_combinado['Grupo'] = [ pass_id[:4] for pass_id in df_combinado['PassengerId']]


df_combinado["total_gastos"] = df_combinado[['RoomService','FoodCourt' , 'ShoppingMall' , 'Spa' , 'VRDeck']].sum(axis=1)

#%%
def split_cabin(row): #return deck - side
    cabin = row['Cabin']
    if isinstance(cabin, str) and not cabin == '':   # transformar Cabin em deck/num/side
        aux = cabin.split('/')
        
        return pd.Series([aux[0] , aux[2]])
    else:
        return pd.Series([None , None])
    
    
import pandas as pd

df = df_combinado
x = pd.DataFrame()
x['Cabin']= df['Cabin']
x[['side', 'deck']] = df.apply(split_cabin, result_type='expand' , axis=1)





#%%
#verificar se todos da mesma familia tem o mesmo destino e home
df = df_combinado

for i, row in df.iterrows():
    
    astr = row['Cabin']
    if isinstance(astr, str) and not astr == '':   # transformar Cabin em deck/num/side
        aux = astr.split('/')
        df.at[i , 'Deck'] = aux[0]
        df.at[i , 'Side'] = aux[2]
        
    #Extrair o sobrenome
    astr = row['Name']
    if isinstance(astr, str) and not astr == '':   # transformar Cabin em deck/num/side
        aux = astr.split()
        df.at[i , 'Sobrenome'] = aux[-1]

total_pessoas_familia = df_combinado.groupby(['Sobrenome' ]).agg({'Sobrenome' : 'count' })
print("Quantidade de pessoas da família")
print(total_pessoas_familia.head)

total_homes_diferentes_por_familia = df_combinado.groupby(['Sobrenome' ])['HomePlanet'].nunique()
print("Valores unicos de HomePlanet por familia")
print(total_homes_diferentes_por_familia.head)

total_destination_diferentes_por_familia = df_combinado.groupby(['Sobrenome' ])['Destination'].nunique()
print("Valores unicos de Destination por familia")
print(total_destination_diferentes_por_familia.head)

df["total_gastos"] = df[['RoomService','FoodCourt' , 'ShoppingMall' , 'Spa' , 'VRDeck']].sum(axis=1)

aux = pd.DataFrame()
aux['CryoSleep'] = df['CryoSleep']
aux['ComGastos'] = np.where(df['total_gastos'] == 0, True, False)

aux = aux.groupby(['ComGastos' , 'CryoSleep']).agg({'CryoSleep':'count'})
print("Valores unicos de VIP por familia")
print(aux.head)






a = df_combinado.groupby(['Sobrenome' , 'HomePlanet' ]).agg({'HomePlanet' : 'count' , 'Destination': 'count'})    
#%%

df_agreg = df_combinado.groupby(['Grupo' ]).agg('count')
#%%

dict_familias_home = dict()
dict_familias_dst = dict()
dict_familias_size = dict()

for i, row in df_combinado.iterrows():
    
    astr = row['Cabin']
    if isinstance(astr, str) and not astr == '':   # transformar Cabin em deck/num/side
        aux = astr.split('/')
        df_combinado.at[i , 'Deck'] = aux[0]
        df_combinado.at[i , 'Side'] = aux[2]
        
    #Extrair o sobrenome
    astr = row['Name']
    if isinstance(astr, str) and not astr == '':   # transformar Cabin em deck/num/side
        aux = astr.split()
        df_combinado.at[i , 'Sobrenome'] = aux[-1]
        
        #montar um map com sobrenome : HomePlanet
        aHome = row['HomePlanet']
        aDestination = row['Destination']
        
        if isinstance(aHome, str) and not aHome == '': 
            dict_familias_home[aux[-1]] = aHome
        if isinstance(aDestination, str) and not aDestination == '': 
            dict_familias_dst[aux[-1]] = aDestination
          
    if aux[-1] in dict_familias_size:
        dict_familias_size[aux[-1]] = dict_familias_size[aux[-1]] + 1
    else:
        dict_familias_size[aux[-1]] = 1
        
            
    if row.total_gastos > 0:   #quem tem gastos não pode estar em Cryo
        df_combinado.at[i , 'CryoSleep'] = False        
        
    if row.CryoSleep:   # se é Cry não tem gastos
        df_combinado.at[i , 'RoomService'] = 0
        df_combinado.at[i , 'FoodCourt'] = 0
        df_combinado.at[i , 'ShoppingMall'] = 0
        df_combinado.at[i , 'Spa'] = 0
        df_combinado.at[i , 'VRDeck'] = 0

#%%
#Mesma familia/sobrenome tem o mesmo HomePlanet e Destination

for i, row in df_combinado.iterrows():
    
    aHome = row['HomePlanet']
    aName = row['Sobrenome']
    aDest = row['Destination']
    if  pd.isnull(aHome):
        if aName in dict_familias_home:  
            df_combinado.at[i , 'HomePlanet'] = dict_familias_home[aName]
            
    if  pd.isnull(aDest):
        if aName in dict_familias_dst:  
            df_combinado.at[i , 'Destination'] = dict_familias_dst[aName]
            
    # incluir feature com tamanho da família
    if aName in dict_familias_size:
        df_combinado.at[i , 'FamilySize'] = dict_familias_size[aName]
    else:
        df_combinado.at[i , 'FamilySize'] = 1
        

df_combinado['HomePlanet'].fillna((df_combinado['HomePlanet'].mode()[0]), inplace = True)           #usar o que mais aparece
df_combinado['Destination'].fillna((df_combinado['Destination'].mode()[0]), inplace = True)        

print(df_combinado.isnull().sum())  
#%%
#ver correlacao entre o tamanho da familia e ser transportado

aux = df_combinado.iloc[:df_treino.shape[0]]
plt.figure(figsize = (12, 4))
sns.countplot(data = aux, x = 'FamilySize', hue = 'Transported', palette = 'viridis')
plt.legend(loc = 'upper right', title = 'Transported')
plt.show()

#nao tem correlacao
#%%
#ver correlacao entre o CryoSleep e ser transportado

aux = df_combinado.iloc[:df_treino.shape[0]]
plt.figure(figsize = (12, 4))
sns.countplot(data = aux, x = 'CryoSleep', hue = 'Transported', palette = 'viridis')
plt.legend(loc = 'upper right', title = 'Transported')
plt.show()

#tem correlação

 
#%%

#Checking correlation between VIP and total_gastos:
plt.figure(figsize = (8, 4))
sns.boxplot(y = df_combinado.VIP, x = df_combinado.total_gastos, orient = 'h', showfliers = False, palette = 'gist_heat')
plt.ylabel('VIP')
#plt.yticks([0,1,2], ['First Class','Second Class', 'Third Class'])
plt.show()

#%%
#existe correlação. ACima de 1000 é VIP=True
df_combinado['VIP'] = df_combinado.apply(
    lambda row: row['VIP'] if not np.isnan(row['VIP']) else ( True if row['total_gastos'] > 1000  else False),
    axis=1
)

 

#%%
df_combinado.to_csv('C:/Users/bergmann/OneDrive/Python/spaceship/saida.csv')

#%%


#%% Simplificaçoes que podem ser melhoradas em uma nova tentativa

#remover os gastos unitarios

df_combinado.drop([ 'ShoppingMall' , 'FoodCourt' , 'RoomService' , 'Spa' , 'VRDeck'] , axis=1 , inplace=True)

#identificadores
df_combinado.drop([ 'PassengerId' , 'Sobrenome' , 'Name'] , axis=1 , inplace=True)

print(df_combinado.isnull().sum()) 
print(((df_combinado.isnull().sum()/df_combinado.isnull().count())*100).round(2))

#%%


#primeira aproximacao. Usar as medias gerais de cada atributo para preencher os vazios
df_combinado['CryoSleep'].fillna((df_combinado['CryoSleep'].mode()[0]), inplace = True)
df_combinado['Age'].fillna((df_combinado['Age'].mean()), inplace = True)
df_combinado['Side'].fillna((df_combinado['Side'].mode()[0]), inplace = True)
df_combinado['Deck'].fillna((df_combinado['Deck'].mode()[0]), inplace = True)


df_combinado.drop([ 'Cabin' ] , axis=1 , inplace=True)

print(df_combinado.isnull().sum()) 
print(((df_combinado.isnull().sum()/df_combinado.isnull().count())*100).round(2))


#%%


print(df_combinado.dtypes)
#transfromando para inteiros
df_combinado['CryoSleep'] = df_combinado['CryoSleep'].astype(int)
df_combinado['VIP'] = df_combinado['VIP'].astype(int)
df_combinado['Grupo'] = df_combinado['Grupo'].astype(int)





# Variaveis categoricas em string para inteiros (Deck Side ...)
names = df['Deck'].unique()
values = list(range(1, names.size + 1))
df_combinado['Deck']= df_combinado['Deck'].replace(names,values)

names = df_combinado['Side'].unique()
values = list(range(1, names.size + 1))
df_combinado['Side']= df_combinado['Side'].replace(names,values)

names = df['HomePlanet'].unique()
values = list(range(1, names.size + 1))

df['HomePlanet']= df['HomePlanet'].replace(names,values)

names = df_combinado['Destination'].unique()
values = list(range(1, names.size + 1))
df_combinado['Destination']= df_combinado['Destination'].replace(names,values)



#%% Montagem dos dados para modelo
#Scaling the independent features:
from sklearn.preprocessing import StandardScaler

df_combinado_y = df_combinado['Transported']
df_combinado_y = df_combinado_y.iloc[:df_treino.shape[0]] 

df_combinado_x = df_combinado.drop(['Transported'] , axis=1 , inplace=False)
df_combinado_x = df_combinado_x.iloc[:df_treino.shape[0]] 

df_combinado_y = df_combinado_y.astype(int)

print(df_combinado_y.dtypes)

# scaler = StandardScaler()
# scaler.fit(df_combinado_x)
# df_combinado_scaled = scaler.fit_transform(df_combinado_x)
# #retorno de transform é um array do Numpy, e no concat() estamos trabalhando com um dataframe. 
# df_combinado_scaled = pd.DataFrame(data = df_combinado_scaled, columns = df_combinado.keys())

#  = df_combinado.iloc[:df_treino.shape[0]]
# df_modelo_teste = df_combinado.iloc[df_treino.shape[0]:]
# df_y = df_modelo_treino['Transported']
# df_y = df_y.astype(int)

#df_modelo_treino = df_combinado_scaled.iloc[:df_treino.shape[0]]

df_y = df_combinado_y
df_x = df_combinado_x


#%%
SEED = 49
from sklearn.model_selection import train_test_split


x_treino, x_teste, y_treino, y_teste = train_test_split(df_x , df_y)

from sklearn import dummy
clf_dummy = dummy.DummyClassifier(random_state=SEED , strategy = 'most_frequent')
clf_dummy.fit(x_treino , y_treino)
print('score dummy:' + str(clf_dummy.score(x_teste , y_teste)))

from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(random_state=SEED , max_depth=5)
clf_tree.fit(x_treino , y_treino)
print('score DecisionTreeClassifier:' + str( clf_tree.score( x_teste , y_teste)))

from sklearn import svm
clf = svm.SVC(random_state=SEED)
clf.fit(x_treino , y_treino)
print('score SVC:' + str( clf.score( x_teste , y_teste)))



#%%

"""# Random Forest"""
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Creating search parameters for the GridSearchCV
search_parameters = [{'n_estimators': [1000],
                     'criterion': ['gini', 'entropy'],
                     'max_depth': [10, 11, 12],
                     'max_leaf_nodes': [18, 19, 20],
                     'min_samples_leaf': [1],
                     'min_samples_split': [2]}]

#Creating a random forest instance and using GridSearchCV to find the optimal parameters:
rf_cls_CV = RandomForestClassifier(oob_score = True, random_state = 10)

grid = GridSearchCV(estimator = rf_cls_CV, param_grid = search_parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

rf_grid = grid.fit(x_treino , y_treino)

print('Best parameters for random forest classifier: ', rf_grid.best_params_, '\n')

#%%
#Creating a random forest model based on the optimal paramters given by GridSearchCV:
rf_grid_model = RandomForestClassifier(n_estimators = rf_grid.best_params_.get('n_estimators'),
                                       criterion = rf_grid.best_params_.get('criterion'),
                                       max_depth = rf_grid.best_params_.get('max_depth'),
                                       max_leaf_nodes = rf_grid.best_params_.get('max_leaf_nodes'),
                                       min_samples_leaf = rf_grid.best_params_.get('min_samples_leaf'),
                                       min_samples_split = rf_grid.best_params_.get('min_samples_split'),
                                       oob_score = True,
                                       random_state = 10, 
                                       n_jobs = -1)

rf_grid_model = rf_grid_model.fit(x_treino , y_treino)

print('score RandomForestClassifier:' + str( rf_grid_model.score( x_teste , y_teste)))

#%%

df_combinado_x = df_combinado.drop(['Transported'] , axis=1 , inplace=False)
df_combinado_x = df_combinado_x.iloc[df_treino.shape[0]:] 

result = rf_grid_model.predict(df_combinado_x)
df_result = pd.DataFrame()
df_result['Transported'] = result.astype(bool)

df_result.to_csv( 'C:/Users/bergmann/OneDrive/Python/spaceship/ulf.csv' , index=False)






