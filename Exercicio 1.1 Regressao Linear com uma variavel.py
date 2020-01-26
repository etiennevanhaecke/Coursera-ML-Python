# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:11:44 2019

@author: e_van
Exercicio 1.1 do curso de ML de Coursera: Regressao Linear com uma variavel
"""

import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Importacao dos dados do arquivo TXT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
data = np.loadtxt("C:\\Users\\e_van\\IA\\Coursera ML\\machine-learning-ex1\\ex1\\ex1data1.txt", delimiter=',')
m = len(data) #Tamanho do jogo de dados
#Array das variaveis de entrada formada da unica featura tamanho da cidade, precedada da coluna de 1 para representada a constante
X = np.concatenate((np.ones((m, 1)), data[:, 0].reshape(m, 1)), axis = 1)
n = X.shape[1] - 1 #Numero de feature desconsiderando a 1era coluna de 1 que representa a constante da hipotese (equacao)
Y = data[:, 1].reshape(m, 1) #Array da variavel de saida (label) Y

#Se monta um grafico para verificar a relacao entre a variavel de entrada tamanho da cidade
# e a variavel de saida o beneficio do Truck
plt.scatter(data[:, 0], data[:, 1], c='g', marker='+') 

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
def GradientDescent(X, Y, theta, alpha, limite, J):
    numIter = 0
    #H1ero istorizacao do custo J em funcao theta a cada iteracao
    histJ = np.empty((1, 3)) #Para historizar o custo em funcao dos parametros theta
    histJ[0, 0] = theta[0]
    histJ[0, 1] = theta[1]
    histJ[0, 2] = J
        
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
        plt.scatter(range(numIter+1), histJ[:, 2], c='y', marker='.')

        #Se desenha a linha de regressao linear em funcao de X e da hipotese configurada por theta
        plt.scatter(data[:, 0], data[:, 1], c='g', marker='+') 
        plt.plot(X[:, 1], np.dot(X, theta))
        
        if np.abs(J-histJ[-2, 2]) < limite:
            break; #Saida da descida do gradient ja que ela encontrou um pomnto baixo
    
    return(numIter, J, theta, histJ)
    
numIter, J, theta, histJ = GradientDescent(X, Y, theta, alpha, limite, J)
print("Gradient descent finalizado depois de {0} iteracoes: Custo J esta {1} e os theta {2}".format(numIter, J, theta))

#Se desenha o grafico do custo J em funcao do numero de iteracao do gradient descendente
print("Evo do custo em funcao iteracao do gradient descent")
plt.scatter(range(numIter+1), histJ[:, 2], c='y', marker='.')

#Se desenha a linha de regressao linear em funcao de X e da hipotese configurada por theta
print("Regressao Linear otimizada no meio dos pontos do jogo de dados")
plt.scatter(data[:, 0], data[:, 1], c='g', marker='+') 
plt.plot(X[:, 1], np.dot(X, theta))

#Calculo dos parametros otimizados com a equacao normal montada considerando que o derivado da funcao custo esta 0
termo1 = np.linalg.inv(np.dot(X.T, X)) #Inverso da matrice obtida com o produto matricial da transposicao de X com X
termo2 = np.dot(X.T, Y) #Produto matricial da transposicao de X com Y
thetaNorm = np.dot(termo1, termo2) #Produto matricial dos 2 membros encima

#Estimativa de lucro para diferentes tamanho de habitantes (NAO ESQUECER DE INICIAR O ARRAY DAS VARIAVEIS DE ENTRADA COM 1 QUE REPRESENTA A CONSTANTE)
pred1 = np.dot(np.array([1, 35]), theta)          #Previsao com os parametros obtidos com o gradient descent para uma cidade de 35 mil habitantes
pred1Norm = np.dot(np.array([1, 35]), thetaNorm)  #Previsao com os parametros obtidos com a equacao normal para uma cidade de 35 mil habitantes
pred2 = np.dot(np.array([1, 70]), theta)          #Previsao com os parametros obtidos com o gradient descent para uma cidade de 70 mil habitantes
pred2Norm = np.dot(np.array([1, 70]), thetaNorm)  #Previsao com os parametros obtidos com a equacao normal para uma cidade de 70 mil habitantes
print("pred1: {0} - pred1Norm: {1} - pred2: {2} - pred2Norm: {3}".format(pred1, pred1Norm, pred2, pred2Norm))
