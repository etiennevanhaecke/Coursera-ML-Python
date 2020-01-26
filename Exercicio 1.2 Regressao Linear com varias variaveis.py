# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:23:00 2019

@author: e_van
Exercicio 1.2 do curso de ML de Coursera: Regressao Linear com varias variaveis
"""

import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Importacao dos dados do arquivo TXT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
data = np.loadtxt("C:\\Users\\e_van\\IA\\Coursera ML\\machine-learning-ex1\\ex1\\ex1data2.txt", delimiter=',')
m = len(data) #Tamanho do jogo de dados
#Array das variaveis de entrada formada das duas feature tamanho e numero quarto da casa (se adicionara a coluna 1 do bias depois da normalizacao destas variaveis de entrada)
X=data[:, :-1].copy()
Y = data[:, -1].reshape(m, 1) #Array da variavel de saida (label) Y
n = X.shape[1] #numero de feature do jogo de dados

def Normalizacao(X, n): 
    histMedia = list() #Historisacao da media de cada feature
    histStd = list() #Historisacao do desvio standard de cada feature
    for i in range(n):
        histMedia.append(np.mean(X[:, i])) #Calculo e historizacao da media da feature
        histStd.append(np.std(X[:, i])) #Calculo e historizacao do desvio padrao da feature
        X[:, i] = (X[:, i] - histMedia[i]) / histStd[i] #Normalizacao da featura substraindo a media e dividindo pelo desvio padrao
    return (X, histMedia, histStd)

X, histMedia, histStd = Normalizacao(X, n) #Normalizacao

#Se adiciona a coluna de 1 que representa a constante (o bias da equaacao linear)
X = np.concatenate((np.ones((m, 1)), X), axis=1)

theta = np.zeros(n+1).reshape(n+1 ,1) #Inicializacao a 0 do array dos parametros das variaveis de entrada, considerando a+ o param da constante
alpha = 0.02 #Fator de aprendizagem
limite = 0.00001 #Variacao em valor absoluto de baixa da funcao custo J entre 2 iteracoes do gradient descent para finali\ar este gradient

#Calculo do custo da hipotese configurada com theta a minimizar para achar os param theta otimos
def CalculoCusto(X, Y, theta):
    #Multiplicacao matricial da hipotese para chegar a predicao com as variaveis de entrada X e os parametros atuais theta
    H = np.dot(X, theta) #Hipotese de calculo da predicao para todo o jogo de dados
    J = 1/(2*m)*np.sum(np.square(H-Y)) #Custo do erro desta predicao H contra o real Y
    return J

#Test da funcao CalculoCusto com os valores iniciais de theta a 0
numIter = 0
J = CalculoCusto(X, Y, theta)
print("Custo inicial {0} com os param theta inicializados a 0".format(J))

#Definidao da funcao de gradient descent que vai otimizar passo a passo os parametros theta
def GradientDescent(X, Y, theta, alpha, limite, J, n):
    numIter = 0
    #H1ero istorizacao do custo J em funcao theta a cada iteracao
    histJ = np.empty((1, n+2)) #Para historizar o custo em funcao dos parametros theta
    for i in range(n+1):
        histJ[0, i] = theta[i]    
    histJ[0, -1] = J
        
    while True:
        print("Iteracao {0}: Custo J esta {1} e os theta {2}".format(numIter, J, theta))
        numIter += 1 #Incremento do numero de iteracao
        
        #Ajuste dos prametros em funcao do derivado da funcao custo
        H = np.dot(X, theta) #Calculo da previsao
        erro = np.dot(X.T, (H-Y)) #Equivalente de somar o erro de todas as linhas do jogo de dado
        gradient = alpha / m * erro
        theta -= gradient
        
        J = CalculoCusto(X, Y, theta) #Calculo do custo com os novos parametros theta
        histJ = np.vstack((histJ, np.hstack((theta.T.ravel(), J)))) #Historisacao do custo por iteracao
        
        #Se desenha o grafico do custo J em funcao do numero de iteracao do gradient descendente
        plt.scatter(range(numIter+1), histJ[:, -1], c='y', marker='.')
        
        if np.abs(J-histJ[-2, -1]) < limite:
            break; #Saida da descida do gradient ja que ela encontrou um pomnto baixo
    
    return(numIter, J, theta, histJ)
    
numIter, J, theta, histJ = GradientDescent(X, Y, theta, alpha, limite, J, n)
print("Gradient descent finalizado depois de {0} iteracoes: Custo J esta {1} e os theta {2}".format(numIter, J, theta))

#Calculo dos parametros otimizados com a equacao normal montada considerando que o derivado da funcao custo esta 0
termo1 = np.linalg.inv(np.dot(X.T, X)) #Inverso da matrice obtida com o produto matricial da transposicao de X com X
termo2 = np.dot(X.T, Y) #Produto matricial da transposicao de X com Y
thetaNorm = np.dot(termo1, termo2) #Produto matricial dos 2 membros encima
print("Com a equacao normal, os parametros theta estao {0}".format(thetaNorm))

#Predicao para uma casa de 1650 metros quadrados e 3 quartos
XTest = np.array([1650, 3])
XTestNorm = (XTest - np.array(histMedia)) / np.array(histStd)  
XTestNorm = np.hstack((np.ones(1), XTestNorm))
print("Previsao de preco de uma cada de 1650m2 e 3 quartos com theta: {0}".format(np.dot(XTestNorm, theta)))
print("Previsao de preco de uma cada de 1650m2 e 3 quartos com thetaNorm: {0}".format(np.dot(XTestNorm, thetaNorm)))

