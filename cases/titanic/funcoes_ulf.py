# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import random

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 1234
random.seed(SEED)


#%%

def grafico_violino(valores, diagnostico , inicio, fim):
    
    plt.figure(figsize=(10,10))
    
    dados_plot = pd.concat([diagnostico, valores.iloc[:,inicio:fim]], axis = 1)
    dados_plot = pd.melt(dados_plot, id_vars="diagnostico", var_name="exames", value_name="valores")
    #plotar de um lado os resultados malignos, e do outro os benignos. Para que isso aconteça, passaremos o parâmetro split = True
    #sns.violinplot(x = "exames", y = "valores", hue = "diagnostico", data = dados_plot)
    sns.violinplot(x = "exames", y = "valores", hue = "diagnostico", split = True, data = dados_plot)
    
    #ajustando o grafico. legenda rotacionada
    plt.xticks(rotation = 90)

#%%

def classificar(nome , valores , diag):
    random.seed(SEED)
    
    treino_x, teste_x, treino_y, teste_y = train_test_split(valores, diag)
    
    from sklearn.ensemble import RandomForestClassifier
    
    classificador = RandomForestClassifier(n_estimators = 100 , random_state=SEED)
    
    classificador.fit(treino_x , treino_y)
    
    print(nome + ' score:' + str( classificador.score( teste_x , teste_y)))    
    
    from sklearn.dummy import DummyClassifier

    classificador_dummy = DummyClassifier(strategy = "most_frequent")

    classificador_dummy.fit(treino_x , treino_y)

    print(nome + ' score dummy:' + str( classificador_dummy.score( teste_x , teste_y)))
#%%

def plotar_matriz_correlacao(matriz_correlacao):
    plt.figure(figsize=(17,15))
    sns.heatmap(matriz_correlacao, annot = True, fmt = ".1f")

    
#%%