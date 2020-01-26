# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:52:37 2019

@author: etienne.vanhaecke
Exercicio 3.2 do curso de ML de Coursera: Rede Neuronal com parametros (=pesos ja dados) para deducao digito escrito manualmente por feedforward propagation

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
data = loadmat("C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex3\\ex3\\ex3data1.mat", variable_names=('X', 'y'))
X=data['X']
m, n = X.shape #Numeros de exemplos de fotos de digitos e de feature (intensidade de gris de cada pixel da imagem do digito)
Y=data['y'] #Recuperacao do label (digito de 0 a 9), com 0 representado por 10

theta = loadmat("C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex3\\ex3\\ex3weights.mat", variable_names=('Theta1', 'Theta2'))
theta1 = theta['Theta1'] #Parametros (=pesos) da funcao mapping para passar da camada de entrada formada de 401 features (400 da foto+constante) ate a camada escondida formada de 25 unidades de ativacao
theta2 = theta['Theta2'] #Parametros (=pesos) da funcao mapping para passar da camada escondida formada de 26 features (25 unidades de ativacao+constante) ate a camada output formada das 10 classes de digito (0 a 9)

def Sigmoid(z):
    z = np.array(z) #Caso z seja passado como lista
    return 1 / (1 + np.exp(-z))

#Previsao do digito de cada foto do jogo de dados passado (pode estar um vector com as features de um unico digito
#como pode estar uma matrice com as features de varias fotos, usando os parametros da rede neuronal ja treinada.
def PrevisaoFeedFwd(X, theta1, theta2):
    #Se X esta um vector linha (um unico exemplo de digito com as suas features)
    if X.ndim == 1:
        X = X.reshape(1, len(X)) #Para esta compativel com os calculos seguintes, se passa ele em array de uma linha
    #Numero de registro exemplo de digito no jogo passado
    m = len(X)
    #Se completa o jogo de dados com a coluna de 1 representando o bias (1era coluna)
    X=np.hstack((np.ones((m, 1)), X)) #CUIDADO coluna de 1 em primeira posicao e nao ultima
    #Calculo do valor de entrada a1 de cada uma das 25 unidades de ativacao (UA) da camada escondida
    #por uma multiplicacao matricial de cada feature da camada de entrada pelos pesos de cada UA (theta1)
    a1 = Sigmoid(np.dot(X, theta1.T)) #Se usa o transverse de theta1 para ter os pesos de cada UI (401) como coluna
                             #Cada coluna de a1 vai corresponder ao valor das 25 UA por exemplo de digito
    
    #Se completa os valores das features das UA com a coluna de 1 representando o bias (1era coluna)
    a1=np.hstack((np.ones((m, 1)), a1)) #CUIDADO coluna de 1 em primeira posicao e nao ultima
    #Calculo do valor de entrada a2 de cada uma das 10 classes (digito de 0 a 9) da camada output
    #por uma multiplicacao matricial de cada feature da camada esconida pelos pesos de cada classe (theta2)
    a2 = Sigmoid(np.dot(a1, theta2.T)) #Se usa o transverse de theta2 para ter os pesos de cada UI (26) como coluna
                              #Cada coluna de a2 vai corresponder ao valor das 10 classes por exemplo de digito
    
    return a2, np.argmax(a2, axis=1)+1 #Se retorna para cada exemplo digito do jogo de dados o digito de maior probabilidade.
                                   #Como os parametros theta foram definidos em uma rede de indice comecando por 1, se adiciona 1
                                   #ja que os index de arrays em Python iniciam por 0 (assim para 0 o valor retornado esta 10)
                                   
#Verificacao da acuracidade de cada classe da rede neuronal sobre o jogo de dados inteiro
print('Acuracidade dos 10 preditores sobre o jogo de dados de {0} exemplos '.format(m))
a2, H=PrevisaoFeedFwd(X, theta1, theta2) #Digito previsto para cada linha exemplo de X
compHcomY = np.hstack((H.reshape(m, 1), Y)) #Comparacao contra a classificacao real colocando o previsto e o label em um mesmo array
for i in range(1, 11):
    #print(i)
    tot = len(Y[Y==i])
    #print(tot)
    ok = np.sum(H.reshape(m, 1)[Y==i]==Y[Y==i])
    #print(ok)
    if i == 10: #Subtilidade do jogo com o dogito 0 representado por 10
        dig = 0
    else:
        dig = i
        
    print('Digito {0}: {1} classificados corretamente sobre os {2} do jogo - {3}%'
          .format(dig, ok, tot, np.round(ok/tot*100, 2)))


    