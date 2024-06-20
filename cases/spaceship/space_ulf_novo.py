# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:59:20 2023

@author: bergmann
"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def preprocess_data(df):
    #remover os gastos unitarios
    df["total_gastos"] = df[['RoomService','FoodCourt' , 'ShoppingMall' , 'Spa' , 'VRDeck']].sum(axis=1)
    df.drop([ 'ShoppingMall' , 'FoodCourt' , 'RoomService' , 'Spa' , 'VRDeck'] , axis=1 , inplace=True)
    
    #adicionar campo com o grupo que tem a mesma variacao no PassengerIf XXXX.01 / 02 / 03 ...
    df['Group'] = [ pass_id[:4] for pass_id in df['PassengerId']]
    
    #separar os atributos embutidos na Canin (deck e side)
    df[['Side', 'Deck']] = df.apply(split_cabin, result_type='expand' , axis=1)
   
    #Extrair o Sobrenome de cada nome
    df['Sobrenome'] = df.apply(extract_family_name, result_type='expand' , axis=1)
    
    #Replace categoric data with values
    df['HomePlanet'] = fill_categoric_field_with_value(df['HomePlanet'])
    df['CryoSleep'] = fill_categoric_field_with_value(df['CryoSleep'])
    df['Destination'] = fill_categoric_field_with_value(df['Destination'])
    df['VIP'] = fill_categoric_field_with_value(df['VIP'])
    df['Transported'] = fill_categoric_field_with_value(df['Transported'])
    
    df['Side'] = fill_categoric_field_with_value(df['Side'])
    df['Deck'] = fill_categoric_field_with_value(df['Deck'])
    
    df['Sobrenome'] = fill_categoric_field_with_value(df['Sobrenome'])
    df['Group'] = fill_categoric_field_with_value(df['Group']) 

def plot_corr_matrix(df , columnNames):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if columnNames is None:
        corr_matrix = df.corr()
    else:
        corr_matrix = df[columnNames].corr()
        
    plt.figure(figsize=(17,15))
    sns.heatmap(corr_matrix , annot = True, fmt = ".1f")

def fill_categoric_field_with_value(serie):
    names = serie.unique()
    values = list(range(1, names.size + 1))
    
    #a tabela de valores continha um float(nan) mapeado para um valor inteiro. Solução foi mudar na tabela de valores colocando o None
    nan_index = np.where(pd.isna(names))
    if len(nan_index) > 0 and len(nan_index[0]) > 0:
        nan_index = nan_index[0][0]
        values[nan_index] = None
    #else:
        #print("Não encontrou nan em " + str(names))
        
    return serie.replace(names,values)

def split_cabin(row): #return deck - side
    cabin = row['Cabin']
    if isinstance(cabin, str) and not cabin == '':   # transformar Cabin em deck/num/side
        aux = cabin.split('/')
        
        return pd.Series([aux[0] , aux[2]])
    else:
        return pd.Series([None , None])

def extract_family_name(row): #return deck - side
    name = row['Name']
    if isinstance(name, str) and not name == '':   
        aux = name.split()
        return pd.Series([aux[-1]])
    return pd.Series([None])

def generate_dict_familias_home(df):
    a_dict = dict()
    for i, row in df.iterrows():
        aHome = row['HomePlanet']
        sobrenome = row['Sobrenome']
        if not (pd.isnull(aHome) or pd.isna(aHome)) and not (pd.isnull(sobrenome) or pd.isna(sobrenome)) : 
            a_dict[sobrenome] = aHome
    return a_dict

def get_home_from_familias_dict(row , a_dict):
    if pd.isnull(row['HomePlanet']):
        if row['Sobrenome'] in a_dict:
            return a_dict[row['Sobrenome']]
    return row['HomePlanet']    


def preprocess_data(df):
    #remover os gastos unitarios
    df["total_gastos"] = df[['RoomService','FoodCourt' , 'ShoppingMall' , 'Spa' , 'VRDeck']].sum(axis=1)
    df.drop([ 'ShoppingMall' , 'FoodCourt' , 'RoomService' , 'Spa' , 'VRDeck'] , axis=1 , inplace=True)
    
    #adicionar campo com o grupo que tem a mesma variacao no PassengerIf XXXX.01 / 02 / 03 ...
    df['Group'] = [ pass_id[:4] for pass_id in df['PassengerId']]
    
    #separar os atributos embutidos na Canin (deck e side)
    df[['Side', 'Deck']] = df.apply(split_cabin, result_type='expand' , axis=1)
   
    #Extrair o Sobrenome de cada nome
    df['Sobrenome'] = df.apply(extract_family_name, result_type='expand' , axis=1)
    
    #Replace categoric data with values
    df['HomePlanet'] = fill_categoric_field_with_value(df['HomePlanet'])
    df['CryoSleep'] = fill_categoric_field_with_value(df['CryoSleep'])
    df['Destination'] = fill_categoric_field_with_value(df['Destination'])
    df['VIP'] = fill_categoric_field_with_value(df['VIP'])
    df['Transported'] = fill_categoric_field_with_value(df['Transported'])
    
    df['Side'] = fill_categoric_field_with_value(df['Side'])
    df['Deck'] = fill_categoric_field_with_value(df['Deck'])
    
    df['Sobrenome'] = fill_categoric_field_with_value(df['Sobrenome'])
    df['Group'] = fill_categoric_field_with_value(df['Group']) 
    
    
    
def transform_null_fields(df):
    #fill Nan fields infered by previos analysing
    
    #preencher HomePlanet com usando o existente para o mesmo Sobrenome
    print('Nulos em HomePlanet:' + str( df['HomePlanet'].isnull().sum()))
    aux_dict = generate_dict_familias_home(df)

    df['HomePlanet'] = df.apply(get_home_from_familias_dict , a_dict = aux_dict , result_type='expand' , axis=1)
    null_count = df['HomePlanet'].isnull().sum()
    total_count =  df['HomePlanet'].isnull().count()
    print('Nulos após processamento inicial:' + str( null_count) + ' ' + str( (null_count/total_count *100).round(2) ) + '%')
    #sobraram poucos nulos, colocar o que mais aparece
    df['HomePlanet'].fillna((df['HomePlanet'].mode()[0]), inplace = True)  
    print('Nulos após processamento final:' + str(df['HomePlanet'].isnull().sum()))



    #preencher CryoSleep = True para quem tem gasto 0
    print('Nulos em CryoSleep:' + str( df['CryoSleep'].isnull().sum()))
    df['CryoSleep'] = df.apply(lambda x: x['CryoSleep'] if not pd.isnull(x['CryoSleep']) else (2 if x['total_gastos'] ==0 else x['CryoSleep']) , axis=1)
    null_count = df['CryoSleep'].isnull().sum()
    total_count =  df['CryoSleep'].isnull().count()
    print('Nulos após processamento inicial:' + str( null_count) + ' ' + str( (null_count/total_count *100).round(2) ) + '%')
    #sobraram poucos nulos, colocar o que mais aparece
    df['CryoSleep'].fillna((df['CryoSleep'].mode()[0]), inplace = True)  
    print('Nulos após processamento final:' + str(df['CryoSleep'].isnull().sum()))    
   
    
#read train data
arquivo = 'C:/Users/bergmann/OneDrive/Python/spaceship/train.csv'
df_1  = pd.read_csv(arquivo)
tamanho_df_treino = df_1.shape[0]
df_1.head()

#read test data. In order to aply transformations in train data, is necessary to do the same transformations in the test data
arquivo = 'C:/Users/bergmann/OneDrive/Python/spaceship/test.csv'
df_2  = pd.read_csv(arquivo)
#df_teste_pass_id = df_2['PassengerId']

#combinar arquivos de treino e teste
df = pd.concat( [df_1 , df_2] , axis=0 , ignore_index=True)

df.head(10)

#%%

print("******************  Initial Data  *********************")

print("---- dataframe sizes -----")
print("Train data:" + str(df_1.shape))
print("Test data:" + str(df_2.shape))
print("Combined data:" + str(df.shape))


print("---- Total Null data -----")
print(df.isnull().sum())
print("---- Data types -----")
print(df.dtypes)

preprocess_data(df)
transform_null_fields(df)
print(df.head)

print("\n\n******************  Data after preprocessing  *********************")
print("---- Total Null data -----")
print(df.isnull().sum())
print("---- Data types -----")
print(df.dtypes)

#%%

#tentar achar os names/sobrenomes que estão null olhando o grupo ou cabine

aux = df.loc[df['Sobrenome'].isna() | df['Group'].isna() | df['Cabin'].isna()]
a = aux.drop(['PassengerId' , 'Destination' , 'Transported' , 'HomePlanet' , 'CryoSleep' , 'Side' , 'Deck' , 'Age' , 'VIP' , 'Name' , 'total_gastos'  ] , axis=1)

display(aux.head(50)) #nao existe um caso em que os dois são nulos
 
       

aux = a.groupby(['Group' ])['Cabin'].nunique()
b=aux.index
print(aux.index)


print("Valores unicos de Sobrenome por Group")
display(aux.value_counts())

