# -*- coding: utf-8 -*-

# lib/funcoes_ulf.py

"""Provide several general ML functions.

This module allows the user to reuse common functions allow ML projects.

The module contains the following functions:

- 'plot_boxplot_for_variables - Plot a boxplot for all variables in variables_list.
- 'def search_for_categorical_variables' - Identify how many unique values exists in each column from df.
- 'plot_frequencias_valores_atributos' - Plot the frequency graphic for the attribute values for each variable in lista_atributos.
- 'plot_correlation_heatmap' = Plot the correlation betwenn pairs of continuos variables.
- 'def analyse_correlation_continuos_variables' - Analyse and plot the correlation betwenn pairs of continuos variables.
- 'analyse_plot_correlation_categorical_variables' - Analyse and plot the correlation betwenn pairs of categorical variables. 

@author: ulf Bergmann

"""


import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2

import pingouin as pg

# %%

def plot_boxplot_for_variables(df , variables_list):
    """Plot a boxplot for all variables in variables_list.
    
    Can be used to verify if the variables are in the same scale

    Examples:
        >>> plot_boxplot_for_variables(df, ['va1' , 'var2' , 'var3'])
        return None

    Args:
        df (DataFrame): DataFrame to be analysed.
        variables_list (list): variable list.

    Returns:
        (None):

    """   
    df_filtered = df[variables_list]

    plt.figure(figsize=(10,7))
    sns.boxplot(x='variable', y='value', data=pd.melt(df_filtered))
    plt.ylabel('Values', fontsize=16)
    plt.xlabel('Variables', fontsize=16)
    plt.show()
    
    return

def search_for_categorical_variables(df):
    """Identify how many unique values exists in each column.

    Parameters:
        df (DataFrame): DataFrame to be analysed.

    Returns:
        cat_stats (DataFrame): Result DataFrame with
        
            - Coluna => the variable name
            - Valores => list os values
            - Contagem de Categorias => count of unique values

    """
    cat_stats = pd.DataFrame(
        columns=['Coluna', 'Valores', 'Contagem de Categorias'])
    tmp = pd.DataFrame()

    for c in df.columns:
        tmp['Coluna'] = [c]
        tmp['Valores'] = [df[c].unique()]
        tmp['Contagem de Categorias'] = f"{len(list(df[c].unique()))}"

        cat_stats = pd.concat([cat_stats, tmp], axis=0)
    return cat_stats


def plot_frequencias_valores_atributos(df, lista_atributos):
    """
    Plot the frequency graphic for the attribute values for each variable in lista_atributos.

    Parameters
    ----------
    df : DataFrame to be analysed

    lista_atributos : variable list 

    Returns
    -------
    None.

    """
    plt.figure(figsize=(15, 45))
    for i in enumerate(lista_atributos):
        plt.subplot(12, 3, i[0]+1)
        grafico = sns.barplot(x=i[1], y=i[1],
                              data=df, estimator=lambda x: len(x) / len(df) * 100)
        grafico.set(ylabel="Percent")




def plot_correlation_heatmap(df, lista_variaveis ):
    """
    Plot the correlation betwenn pairs of continuos variables.

    Parameters
    ----------
    df : DataFrame to be analysed

    lista_variaveis: continuos variables list


    Returns
    -------
    None

    """
    cv_df = df[lista_variaveis]

    # metodos: 'pearson', 'kendall', 'spearman' correlations.
    corr_matrix = cv_df.corr(method='pearson')

    fig = plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, annot_kws={'size': 15} , cmap="Blues")
    plt.title("Correlation Heatmap")
    #fig.tight_layout()

    fig.show()

def analyse_correlation_continuos_variables(df, lista_variaveis , quant_maximos):
    """
    Analyse and plot the correlation betwenn pairs of continuos variables.

    Parameters
    ----------
    df : DataFrame to be analysed

    lista_variaveis: continuos variables list

    quant_maximos : number of maximum values


    Returns
    -------
    top_pairs_df : sorted DataFrame with Variable1 | Variable 2 | Correlation
    corr_matrix : Correlation matrix with p-values on the   upper triangle 


    """
    cv_df = df[lista_variaveis]

    # metodos: 'pearson', 'kendall', 'spearman' correlations.
    corr_matrix = cv_df.corr(method='pearson')

    # Gera uma matriz de correlação onde a parte superior contem os p-valor
    # da correlação entre as variaveis considerando o nivel de significancia
    matriz_corr_with_pvalues = pg.rcorr(cv_df, method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

    # Get the top n pairs with the highest correlation
    top_pairs = corr_matrix.unstack().sort_values(ascending=False)[
        :len(df.columns) + quant_maximos*2]

    # Create a list to store the top pairs without duplicates
    unique_pairs = []

    # Iterate over the top pairs and add only unique pairs to the list
    for pair in top_pairs.index:
        if pair[0] != pair[1] and (pair[1], pair[0]) not in unique_pairs:
            unique_pairs.append(pair)

    # Create a dataframe with the top pairs and their correlation coefficients
    top_pairs_df = pd.DataFrame(
        columns=['feature_1', 'feature_2', 'corr_coef'])
    for i, pair in enumerate(unique_pairs[:quant_maximos]):
        top_pairs_df.loc[i] = [pair[0], pair[1],
                               corr_matrix.loc[pair[0], pair[1]]]

    return top_pairs_df , matriz_corr_with_pvalues

# =============================================================================
# Analisa a correlacao de variaveis categoricas usando o teste Qui-quadrado.
# Plota o heatmap do p-valor
# relativo ao seguinte teste de hipóteses:
#    H0: variáveis são dependentes
#    H1: Variáveis são independentes.
# ver https://www.analyticsvidhya.com/blog/2021/06/decoding-the-chi-square-test%E2%80%8A-%E2%80%8Ause-along-with-implementation-and-visualization/

# =============================================================================


def analyse_plot_correlation_categorical_variables(df, lista_variaveis):
    """
    Analyse and plot the correlation betwenn pairs of categorical variables. Variables must be not continuos (not float).

    Use the qui-quadrad and p-value for:
        H0: dependent variables
        H1: independent variables

    Parameters
    ----------
    df : DataFrame to be analysed

    lista_variaveis : Variable list

    Returns
    -------
    resultant : Dataframe with all p-values
    lista_resultado_analise : array with Variable1 | Variable 2 | p-value


    """
    resultant = pd.DataFrame(data=[(0 for i in range(len(lista_variaveis))) for i in range(len(lista_variaveis))],
                             columns=list(lista_variaveis), dtype=float)
    resultant.set_index(pd.Index(list(lista_variaveis)), inplace=True)

    # Encontrando os p-valores para as variáveis e formatando em matriz de p-valor
    lista_resultado_analise = []
    for i in list(lista_variaveis):
        for j in list(lista_variaveis):
            if i != j:
                try:
                    chi2_val, p_val = chi2(
                        np.array(df[i]).reshape(-1, 1), np.array(df[j]).reshape(-1, 1))
                    p_val = round(p_val[0], 4)
                    resultant.loc[i, j] = p_val
                    lista_resultado_analise.append([i, j,  p_val])
                except ValueError:
                    print(f"Variavel {j} não é categórica ")
                    return

    fig = plt.figure(figsize=(25, 20))
    sns.heatmap(resultant, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Resultados do teste Qui-quadrado (p-valor)')
    plt.show()

    return resultant, lista_resultado_analise


def fill_categoric_field_with_value(serie, replace_nan):
    """


    Parameters
    ----------
    serie : TYPE
        DESCRIPTION.
    replace_nan : Boolean. True Replace nan values by 'N/A' .

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    names = serie.unique()
    values = list(range(1, names.size + 1))

    # a tabela de valores continha um float(nan) mapeado para um valor inteiro. Solução foi mudar na tabela de valores colocando o None
    nan_index = np.where(pd.isna(names))
    if len(nan_index) > 0 and len(nan_index[0]) > 0:
        nan_index = nan_index[0][0]
        values[nan_index] = None
    # else:
        # print("Não encontrou nan em " + str(names))

    return serie.replace(names, values)


