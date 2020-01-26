# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:26:52 2019

@author: e_van
Exercicio 2.2 do curso de ML de Coursera: Regressao logistica com regularizacao para identificar os micro-chip que nao passam o QA em funcao de 2 testes
"""

import numpy as np
import matplotlib.pyplot as plt

#Importacao dos dados do arquivo TXT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
data = np.loadtxt("C:\\Users\\e_van\\IA\\Coursera ML\\machine-learning-ex2\\ex2\\ex2data2.txt", delimiter=',')
m = len(data) #Tamanho do jogo de dados

X=data[:, :-1] #Variaveis de entrada com o resultado dos 2 testes
n=X.shape[1] #Numero de feature sem considerar o bias

Y=data[:, -1].reshape(m,1) #Label passagem (valor 1) ou nao (valor 0) o QA

aceito=X[Y.ravel()==1] #Jogo de dado dos 2 testes do grupo dos micro-chops que passaram o QA
recusado=X[Y.ravel()==0] #Jogo de dado dos 2 testes do grupo dos micro-chops que NAO passaram o QA

#Se desenha o jogo de dado em funcao dos 2 testes
plt.close('all')
fig1 = plt.figure(figsize=(10, 6))
ax1=fig1.add_subplot(1, 1, 1)
ax1.scatter(x=aceito[:, 0], y=aceito[:, 1], c='k', marker='+', label='ACEITO QA')
ax1.scatter(x=recusado[:, 0], y=recusado[:, 1], c='y', marker='o', label='RECUSADO QA')
plt.legend(loc='best') #Opcao best nao implementada usando legend com plt
plt.xlabel('Nota Test 1')
plt.ylabel('Nota Test 2')
ax1.set_title('Os 2 grupos de micro-chip em funcao do aceite no QA e das notas dos 2 testes')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 6, include_bias=True)
X=poly.fit_transform(X) #Se cria novas features a parte dos 2 testes, montando um polinomial destes testes ate o sexto grau e considerando o bias
n=X.shape[1]-1 #Novo Numero de feature sem considerar o bias


#Parametros das variaveis de entrada inicializada por default a 0 (considerando o param da constante)
theta = np.zeros((n+1, 1))
initial_theta= theta.ravel().copy() #Usado na funcao de otimizacao e deve estar um vector linear de 1 dimensao (n,)
                                    #IMPORTANTE: Pensar a usar copy senao seeria so uma nova referencia sobre o mesmo espaco memoria que theta
def Sigmoid(z):
    z = np.array(z) #Caso z seja passado como lista
    return 1 / (1 + np.exp(-z))

def CalculoCustoReg(theta, X, Y, m, n, lbd):
    theta = theta.reshape((n+1), 1) #Se redimensiona ja que para estar usado na proc de otimizacao, tem que passar theta como vector linear de 1 dimensao (n,)
    H=Sigmoid(np.dot(X, theta)) #Calculo da predicao de classificacao de cada estudante do jogo de dados em funcao das notas X e da parametrizacao theta
    J1 = 1/m * np.sum(( - Y*np.log(H)) - (1-Y)*(np.log(1-H)))
    #Abaixo outra maneira de calcular trocando o uso da funcao soma pela multiplicacao matricial (TALVE MAIS RAPIDO)
    J2 = 1/m * ( - np.dot(Y.T, np.log(H)) - np.dot((1-Y).T, np.log(1-H)))
    #Regularizacao com lambda usando o metodo Ridge
    JReg = J2 + (lbd/(2*m) * np.sum(theta**2))  
    return JReg.ravel() #Se fica com uma unida dimensao para nao ter problema durante a historizacao deste custo na funcao GradientDescent

lbd = 1 #Fator de regularizacao para evitar o overfitting penalisando as hipoteses com maior soma de parametros theta
alpha = 0.1 #Fator de aprendizagem
limite = 0.0000001 #Variacao em valor absoluto de baixa da funcao custo J entre 2 iteracoes do gradient descent para finali\ar este gradient
    
#Test da funcao CalculoCusto com os valores iniciais de theta a 0
numIter = 0
J = CalculoCustoReg(theta, X, Y, m, n, lbd)
print("Custo inicial {0} com os param theta inicializados a 0".format(J))

#Calculo do derivado da funcao custo (theta deve estar em 1era posicao para pode estar chamada pela funcao de otimizacao)
def GradientReg(theta, X, Y,  m, n, lbd):
    theta = theta.reshape((n+1), 1) #Se redimensiona ja que para estar usado na proc de otimizacao, tem que passar theta como vector linear de 1 dimensao (n,)
    H = Sigmoid(np.dot(X, theta)) #Calculo da previsao
    erro = np.dot(X.T, (H-Y)) #Equivalente de somar o erro de todas as linhas do jogo de dado
    #Se regulariza com o metodo Ridge so o gradient dos parametros theta de feature (a constante/bias nao esta a regularizar)
    erro[1:] = erro[1:] + lbd/m*theta[1:]
    gradient = 1 / m * erro
    return gradient

#Definidao da funcao de gradient descent que vai otimizar passo a passo os parametros theta
def GradientDescent(X, Y, theta, alpha, limite, J, n, m, lbd):
    numIter = 0
    #H1ero istorizacao do custo J em funcao theta a cada iteracao
    histJ = np.empty((1, n+2)) #Para historizar o custo em funcao dos parametros theta
    for i in range(n+1):
        histJ[0, i] = theta[i]    
    histJ[0, -1] = J
        
    while True:
        print("Iteracao {0}: Custo J esta {1}".format(numIter, J))
        numIter += 1 #Incremento do numero de iteracao
        
        #Ajuste dos prametros em funcao do derivado da funcao custo
        gradient = GradientReg(theta, X, Y, m, n, lbd)
        #print('-> Gradient {0} a aplicar nos parametros theta.'.format(gradient))
        theta -= alpha*gradient
        
        J = CalculoCustoReg(theta, X, Y, m, n, lbd) #Calculo do custo com os novos parametros theta
        histJ = np.vstack((histJ, np.hstack((theta.T.ravel(), J)))) #Historisacao do custo por iteracao
                
        if np.abs(J-histJ[-2, -1]) < limite:
            break; #Saida da descida do gradient ja que ela encontrou um pomnto baixo
    
    return(numIter, J, theta, histJ)
    
numIter, J, theta, histJ = GradientDescent(X, Y, theta, alpha, limite, J, n, m, lbd)
print("Gradient descent finalizado depois de {0} iteracoes: Custo J esta {1} e os theta {2}".format(numIter, J, theta.ravel()))

#Se desenha o grafico do custo J em funcao do numero de iteracao do gradient descendente
#plt.close()
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.scatter(range(numIter+1), histJ[:, -1], c='y', marker='.', label = 'Custo J por Iteracao')
plt.legend(loc='best') #Opcao best nao implementada usando legend com plt
plt.xlabel('Num. Iteracao')
plt.ylabel('Custo J')
ax1.set_title('Custo J em funcao iteracao')
plt.show()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty="l1", solver='warn', max_iter=1000)
clf.fit(X,Y.ravel())
lassoTheta=clf.coef_.T
print("The regularized theta using lasso regression:\n",lassoTheta)

#Uso de funcoes de optimizacao da biblioteca scipy.optimize
from scipy.optimize import minimize, fmin_tnc, fmin, fmin_bfgs, fmin_ncg, leastsq, fmin_slsqp

Result = minimize(fun = CalculoCustoReg, x0 = initial_theta, args = (X, Y, m, n, lbd),
                  method = 'TNC', jac = GradientReg);
optTheta = Result.x;
optJ = Result.fun
print('Com minimize de scipy.optimize se chega a um custo optJ de {0} e optTheta {1}'.format(optJ, optTheta))

Result = fmin_tnc(func=CalculoCustoReg, x0=initial_theta, args=(X, Y, m, n, lbd), fprime=GradientReg)
tncTheta = Result[0];
print('Com fmin_tnc de scipy.optimize se chega a tncTheta {0}'.format(tncTheta))

#Versao sem passar a funcao de calculo do gradient (parametro fprime), informando o param approx_gradbool com True
Result = fmin_tnc(func=CalculoCustoReg, x0=initial_theta, args=(X, Y, m, n, lbd), approx_grad=True) 
tncTheta = Result[0];
print('Com fmin_tnc de scipy.optimize, SEM PASSAR A FCT GRADIENT, se chega a tncTheta {0}'.format(tncTheta))

def Previsao(X, theta, n):
    theta = theta.reshape((n+1), 1) #Caso nao esta passado como vector de coluna (matrice de 1 coluna)
    H = np.round(Sigmoid(np.dot(X, theta)), 2)
    return H

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def Metrica(X, theta, n, Y, tipoTheta):
    H=Previsao(X, theta, n) #Probabilidade de passar o QA
    H=np.round(H) #Para ter a classificacao 1/0 passando ou nao o QA    
    #compHcomY = np.hstack((H, Y)) #Comparacao contra a classificacao real colocando o previsto e o label em um mesmo array
    #print('compHcomY {0}'.format(compHcomY))
    confMatrix = confusion_matrix(Y, H) #Confusion matrix apresentando em uma matrice de 2X2, com label em linha e predicao em coluna,
                                    #o RN (Right Negative), o FP (False Positive) na PRIMEIRA linha E
                                    #o FN (False Negative), o RP (Rigt Positive) na SEGUNDA linha
    accScore = accuracy_score(Y, H) #ACCURACIDADE: Fracao de exemplos classificados corretamente = (TN+TP) / (TN+FP+FN+TP)
    recScore = recall_score(Y, H) #RACALL: Fracao de exemplos positivos (aceitos na universidade) corretamente classificados = RP / (RP+FN)
    precScore = precision_score(Y, H) #PRECISION: Fracao de exemplos predictos como positivos que realmente estao positivos: = RP / (RP+FP)
    F1Score = f1_score(Y, H) #f1 score is the harmonic mean of recall and precision, with a higher score as a better model

    print('=== confMatrix theta tipo {0} ===\n{1}'.format(tipoTheta, confMatrix))
    print('--> accScore: {0} - recScore: {1} - precScore: {2} - F1Score: {3}'
          .format(accScore, recScore, precScore, F1Score))
    
Metrica(X, theta, n, Y, 'Gradient Descent Regr. Ridge')
Metrica(X, lassoTheta, n, Y, 'Regressao Lasso')
Metrica(X, optTheta, n, Y, 'Otimize Regr. Ridge')
Metrica(X, tncTheta, n, Y, 'fmin_tnc Regr. Ridge')

#Desenho da linha de separacao (decision boundary) entre as 2 classes
numNota=70
rangeNota1 = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), numNota) #numNota pontos do axe x da nota 1
rangeNota2 = np.linspace(np.min(X[:, 2]), np.max(X[:, 2]), numNota) #numNota pontos do axe x da nota 1
HrangeNotas = np.zeros((numNota, numNota))
for i in range(len(rangeNota1)):
    for j in range(len(rangeNota2)):
        HrangeNotas[i, j] = np.round(Previsao(poly.fit_transform(np.vstack((rangeNota1[i], rangeNota2[j])).T), theta, n))
#Se desenha a decision boudary, fazendo cuidado que devido a presenca da constante em 1era posicao, a nota 1 tem o index 1 e a nota 2 o index 2
ax1.contour(rangeNota1, rangeNota2, HrangeNotas, 0, colors='r')





