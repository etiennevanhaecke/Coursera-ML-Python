# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 09:05:11 2019

@author: etienne.vanhaecke
Exercicio 5.1 do curso de ML de Coursera: Regularized Linear Regression

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
data = loadmat("C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex5\\ex5\\ex5data1.mat")

# training set
X, Y = data['X'], data['y']

# cross-validation set
XCv, YCv = data['Xval'], data['yval']

# test set
XTest, YTest = data['Xtest'], data['ytest']

m, n = X.shape
mCv = XCv.shape[0]
mTest = XTest.shape[0]
lbd = 2 #Fator de regularizacao para limitar o overfitting

plt.close('all')
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(111)
ax1.scatter(X, Y, marker='x', c='red')
plt.xlabel('Change in water level (X)')
plt.ylabel('Water flowing out of the dam (Y)')
plt.title('FIGURE 1: DATA')

def FeatureNormalize(X):
    '''
    Funcao de normalizacao das variaveis de entraada (feature)
    Costuma corresponder a substracao da media e a divisao pelo desvio padrao 
    Permite de acelerar a otimizacao dos parametros da funcao objetivo
    FEATURENORMALIZE Normalizes the features in X 
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    avgX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)
    
    XNorm = (X-avgX) / stdX
    
    return XNorm, avgX, stdX

#Se normaliza X para otimizar a busca dos parametros otimos da funcao objetivo
#X, avgX, stdX = FeatureNormailize(X)
#Se adiciona a coluna de 1 a X para representar a constante 
X = np.hstack((np.ones((m ,1)), X) )
#Se inicializa os parametros de peso theta com 1
#Valores iniciais dos pesos a passar a funcao de otimizacao
thetaInit=np.ones((n+1, 1)) #+1 em comparacao ao numero de feature para considerar a constante (=bias)

#IMPORTANTE: Para passar esta funcao como funcao de custo e de grad em um
#otimizador precisa:
# - Ter theta em 1era posicao e depois os outros parametros em qualquer ordem
# - Permitir que theta seja passado como um vector horizontal (uma dimensao) e entretanto precisa redimensionar ele dentro da funcao
# - Retornar o gradient como um vector horizontal (uso flatten ou ravel)
def LinearRegCostFunction(theta, X, Y, m, n, lbd):
    '''
LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
regression with multiple variables
   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
   cost of using theta as the parameter for linear regression to fit the 
   data points in X and y. Returns the cost in J and the gradient in grad
   '''
    #Se passa theta como vector colunar 
    theta=theta.reshape(np.size(theta), 1)
    #Calculo do custo sem regularizacao
    J = 1/(2*m) * np.sum(np.power((np.dot(X, theta) - Y), 2))
    #print('Metodo 1 J: ', J)
    #Calculo da regularizacao em funcao de lambda
    reg=lbd/(2*m) * np.sum(theta[1:]**2) #Inicio do index 1 de theta para nao considerar o peso da constante de index 0
    #print('Metodo 1 Reg: ', reg)
    #Custo regularizado
    JReg = J+reg
    #print('Metodo 1 JReg: ', JReg)

    #Outra solucao, usando outra funcao para o produto matricial e usando o produto matricial para o power 2, achada em
    #https://github.com/MohammadKhalaji/Andrew-NG-Machine-Learning-Assignments-Python/blob/master/ex5/ex5.ipynb
    # Not regularized yet
    ans = np.matmul(X, theta)
    ans = ans - Y
    ans = np.matmul(ans.T, ans)
    ans = (1 / (2 * m)) * ans
    #print('Metodo 2 J: ', ans)
    # Regularization term
    reg = np.matmul(theta[1:, ...].T, theta[1:, ...])
    reg = (lbd / (2 * m)) * reg
    #print('Metodo 2 reg: ', reg)
    ans = (ans + reg).flatten()[0]
    #print('Metodo 2 ans: ', ans)
    
    #Calculo do gradient de cada peso de variavel de entrada (incluindo a constante)
    #OBS: Uso de @ que esta um short cut para a funcao de multiplicacao matricial 
    grad = 1 / m * ((X@theta - Y).T @ X)
    #print('grad SEM regularizacao ', grad)
    reg = lbd/m*theta[1] #Nao se regulariza o gradient do peso de index 0 que corresponde ao bias
    #print('Valor reg da egularizacao de gradient ', reg)
    grad[0, 1:] = grad[0, 1:]+reg #Regularizacao do gradient sem considerar a constante
    #print('Valor do gradient Regularizado ', grad)    
    
    return JReg, grad.ravel()

#Calculo do custo com os param iniciais
JInit, GradInit = LinearRegCostFunction(thetaInit, X, Y, m, n, lbd) 
print('JInit {0} e GradInit {1} Sem normalizacao de X '.format(JInit, GradInit))

#Se desenha a linha desta regressao linear com estes pesos iniciais na figura 1
idxXMin = np.argmin(X[:, 1])
idxXMax = np.argmax(X[:, 1])
YMin = X[idxXMin] @ thetaInit
YMax = X[idxXMax] @ thetaInit  

ax1.plot((X[idxXMin, 1], X[idxXMax, 1]), (YMin, YMax), 'o-b', label='Param iniciais')

from scipy.optimize import minimize

def TrainLinearReg(thetaInit, X, Y, m, n, lbd):    
    '''
TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
regularization parameter lambda
   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
   the dataset (X, y) and regularization parameter lambda. Returns the
   trained parameters theta.
    ''' 
    #Uso de funcoes de optimizacao da biblioteca scipy.optimize
    #Metodo CG
    #Method CG uses a nonlinear conjugate gradient algorithm by Polak and Ribiere, a variant of 
    #the Fletcher-Reeves method described in 5 pp. 120-122. Only the first derivatives are used.
    Result = minimize(method = 'CG', fun = LinearRegCostFunction, x0 = thetaInit, 
                      args = (X, Y, m, n, lbd), jac = True);
    thetaCG = Result.x;
    JCG = Result.fun
    #print('Com metodo de otim CG de scipy.optimize se chega a um custo JOpt de {0}'.format(JCG))

    #Metodo Nelder-Mead
    #Method Nelder-Mead uses the Simplex algorithm 1, 2. This algorithm is robust in many applications.
    #However, if numerical computation of derivative can be trusted, other algorithms using the first 
    #and/or second derivatives information might be preferred for their better performance in general.
    #Result = minimize(method = 'Nelder-Mead', fun = LinearRegCostFunction, x0 = thetaInit, 
    #                  args = (X, Y, m, n, lbd), jac = True);
    #thetaNM = Result.x;
    #JNM = Result.fun
    #print('Com metodo de otim Nelder-Mead de scipy.optimize se chega a um custo JOpt de {0}'.format(JNM))

    #Metodo TNC
    #Method TNC uses a truncated Newton algorithm 5, 8 to minimize a function with variables subject to bounds. 
    #This algorithm uses gradient information; it is also called Newton Conjugate-Gradient. It differs from the
    #Newton-CG method described above as it wraps a C implementation and allows each variable to be given upper and lower bounds.
    #Result = minimize(method = 'TNC', fun = LinearRegCostFunction, x0 = thetaInit, 
    #                  args = (X, Y, m, n, lbd), jac = True);
    #thetaTNC = Result.x;
    #JTNC = Result.fun
    #print('Com metodo de otim TNC de scipy.optimize se chega a um custo JOpt de {0}'.format(JTNC))
                      
    #Se usa os pesos retornados pelo otimizador de menor custo J
    thetaOpt = thetaCG
    JOpt = JCG
    '''if (JCG < JNM and JCG < JTNC):
        thetaOpt = thetaCG
        JOpt = JCG
    elif (JNM < JTNC):
        thetaOpt = thetaNM
        JOpt = JNM
    else:
        thetaOpt = thetaTNC
        JOpt = JTNC
    '''
    return thetaOpt, JOpt #Retorno dos parametros otimizados para o modelo definido em LinearRegCostFunction
  
#Treinamento do modelo deixando a regularizacao lbd a 0 (nao tem ganho com um modelo simple de 2 variaveis)
thetaOpt, JOpt = TrainLinearReg(thetaInit.ravel(), X, Y, m, n, lbd)
print('Pesos theta otimizados com o modelo simple sem regularizacao ', thetaOpt)

#Se desenha a linha do modelo com os parametros theta otimizados
YMin = X[idxXMin] @ thetaOpt
YMax = X[idxXMax] @ thetaOpt  

ax1.plot((X[idxXMin, 1], X[idxXMax, 1]), (YMin, YMax), 'o-r', label='Param otimizados')
ax1.legend(loc='best')

def LearningCurve(thetaInit, X, Y, m, n, lbd, XCv, YCv, mCv):
    '''
LEARNINGCURVE Generates the train and cross validation set errors needed 
to plot a learning curve
   [error_train, error_val] = ...
       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
       cross validation set errors for a learning curve. In particular, 
       it returns two vectors of the same length - error_train and 
       error_val. Then, error_train(i) contains the training error for
       i examples (and similarly for error_val(i)).

   In this function, you will compute the train and test errors for
   dataset sizes from 1 up to m. In practice, when working with larger
   datasets, you might want to do this in larger intervals.
    '''
   
    error_train = np.array([]) #Vector do Custo J do jogo de treinamento em funcao do tamanho deste jogo
    error_val = np.array([])   #Vector do Custo J do jogo de cross-validacao em funcao do tamanho do jogo de treinamento

    XCv = np.hstack((np.ones((mCv, 1)), XCv))   #Se adiciona a coluna de 1 em 1era posicao do jogo de Cross Validacao
    #Loop sobre o numero de exemplo do jogo de treinamento para treinar o modelo
    #de regressao linear para cada sub-set deste jogo (de 1 exemplo ate todos os exemplos)
    for i in range(m):
        print('Custos com o jogo de treinamento reduzido a {0} exemplo'.format(i+1))
        #Treinamento do modelo sobre o subset dos i registros de X com retorno do custo com param otim para o jogo de treinamento
        theta, J = TrainLinearReg(thetaInit.ravel(), X[0:i+1], Y[0:i+1], m=i+1, n=n, lbd=lbd)
        print('-> J Treinamento: ', J)
        error_train = np.append(error_train, J) #Info do vector do custo do jogo de treinamento
        #Calculo do custo para o jogo completo de validacao cross usando os param otimizados encima com o jogo parcial de treinamento
        J, gradNaoUsado = LinearRegCostFunction(theta, XCv, YCv, mCv, n, lbd=0) #Nao aplicar regularizacao sobre o jogo de validacao cross
        print('-> J Validacao: ', J)
        error_val = np.append(error_val, J) #Info do vector do custo do jogo de treinamento
        
    return error_train, error_val 

#Verificacao da evo do custo do jogo de treinamento e do jogo de validacao
#em funcao do numero de exemplo do jogo de treinamento
error_train, error_val = LearningCurve(thetaInit, X, Y, m, n, lbd, XCv, YCv, mCv)

#Desenho da curva de aprendizagem da regressao linear simple
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111)
ax2.plot(range(1, m+1), error_train, 'o-r', label='Jogo Treinamento')
ax2.plot(range(1, m+1), error_val, 'x-b', label='Jogo Validacao')
plt.xlabel('Number of training exemples')
plt.ylabel('Error')
plt.title('FIG 2: Linear Regression Learning Curve')
ax2.legend(loc='best')

def PolyFeatures(X, p, m):
    '''
POLYFEATURES Maps X (1D vector) into the p-th power
   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
   maps each example into its polynomial features where
   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    '''
    X = X.reshape(m, 1)
    for i in range(p):
        X = np.hstack((X, (X[:, i] * X[:, 0]).reshape(m, 1) ))
        
    return X


def PlotFit(X, Y, m, idxXMin, idxXMax, lbd):
    '''
PLOTFIT Plots a learned polynomial regression fit over an existing figure.
Also works with linear regression.
   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
   fit with power p and feature normalization (mu, sigma).
    '''
    #Se monta um novo jogo de treinamento usando um polinomial de 8 da unica variavel de entrada
    X = PolyFeatures(X[:, 0:], 8, m) #Se desconsidera a constante que esta na 1era coluna
    #Visto a grande diferenca de valor entre as features polinomiais, se aplica a normalizacao
    X, avgX, stdX = FeatureNormalize(X)
    n = X.shape[1] # Novo numero de parametros (1 valor original+8 polinomais deste valor de **2 a **8)
    #Se volta a adicionar a coluna de 1 em 1era posicao para ter a constante da regressao linear
    X = np.hstack((np.ones((m ,1)), X))
    #Se inicializa os parametros de peso theta com 1
    #Valores iniciais dos pesos a passar a funcao de otimizacao
    thetaInit=np.ones((n+1, 1)) #+1 em comparacao ao numero de feature para considerar a constante (=bias)
    #Treinamento do modelo deixando a regularizacao lbd a 0 (nao tem ganho com um modelo simple de 2 variaveis)
    thetaOpt, JOpt = TrainLinearReg(thetaInit.ravel(), X, Y, m, n, lbd)
    print('Pesos theta {0} otimizados com o modelo polinomial de 8 de regularizacao {1}'.format(thetaOpt, lbd))
    print('-> Custo JOpt de X completo de ', JOpt)
    thetaOpt = thetaOpt.reshape(n+1, 1) #Se redimensiona em matrice de 1 coluna para pode aplicar o produto matricial com X

    #Se desenha a linha do modelo com os parametros theta polinomais otimizados
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    #Valor original do parametro de entrada presente na coluna 2, pensando a denornalizar X
    ax3.scatter(X[:, 1]*stdX[0]+avgX[0], Y, marker='x', c='red') 
    plt.xlabel('Change in water level (X)')
    plt.ylabel('Water flowing out of the dam (Y)')
    plt.title('FIGURE 3: FIT COM OS PARAM OTIMIZADOS E DE FATOR REGUL {0}'.format(lbd))
    #Se define um intervalo para plot em funcao do min-5 e max+5 do valor original da variavel de entrada.
    #Como dentro do array X, este valor presente na segunda coluna (a 1era esta a constante 1), esta nornalizada,
    #precisa demormalizar ela, pensando quea a media e o desvio padrao da nornalizacao da variavel de entrada
    #estao na prineira coluna (index 0) dos arrays stdX e avgX (a normalizacao de X com polinome de 8 foi feita
    # antes de adicionar a coluna de 1).
    plotX = np.arange(start=X[idxXMin, 1]*stdX[0]+avgX[0]-5, stop=X[idxXMax, 1]*stdX[0]+avgX[0]+5)
    #Se calcula a predicao Y de cada valor do intervalo plotX, aplicando a adicao dos polinomes ate 8,
    #normalizando, adicionando a coluna da constante 1 e calculando com os parametros otimizados
    mPlot = plotX.shape[0]
    plotY = np.hstack((np.ones((mPlot, 1)), (PolyFeatures(plotX, 8, mPlot) - avgX)/stdX)) @ thetaOpt

    ax3.plot(plotX, plotY, 'o--b', label='Fit com poli nao regul', alpha=0.2)
    ax3.legend(loc='best')

X, Y = data['X'], data['y'] #Se reinicia X e Y a parte do jogo originak

PlotFit(X, Y, m, idxXMin, idxXMax, lbd)
        
def ValidationCurve(X, Y, m, lbd, XCv, mCv, YCv):
    '''
VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
   [lambda_vec, error_train, error_val] = ...
       VALIDATIONCURVE(X, y, Xval, yval) returns the train
      and validation errors (in error_train, error_val)
      for different values of lambda. You are given the training set (X,
       y) and validation set (Xval, yval).
    '''
    #Se monta um novo jogo de treinamento usando um polinomial de 8 da unica variavel de entrada
    X = PolyFeatures(X[:, 0:], 8, m) #Se desconsidera a constante que esta na 1era coluna
    #Visto a grande diferenca de valor entre as features polinomiais, se aplica a normalizacao
    X, avgX, stdX = FeatureNormalize(X)
    n = X.shape[1] # Novo numero de parametros (1 valor original+8 polinomais deste valor de **2 a **8)
    #Se volta a adicionar a coluna de 1 em 1era posicao para ter a constante da regressao linear
    X = np.hstack((np.ones((m ,1)), X))
    #Se inicializa os parametros de peso theta com 1
    #Valores iniciais dos pesos a passar a funcao de otimizacao
    thetaInit=np.ones((n+1, 1)) #+1 em comparacao ao numero de feature para considerar a constante (=bias)

    #Se monta um novo jogo de Cross Validacao usando um polinomial de 8 da unica variavel de entrada
    XCv = PolyFeatures(XCv[:, 0:], 8, mCv) #Se desconsidera a constante que esta na 1era coluna
    #Visto a grande diferenca de valor entre as features polinomiais, se aplica a normalizacao
    XCv, avgXCv, stdXCv = FeatureNormalize(XCv)

    #Verificacao da evo do custo do jogo de treinamento e do jogo de validacao
    #em funcao do numero de exemplo do jogo de treinamento
    error_train, error_val = LearningCurve(thetaInit, X, Y, m, n, lbd, XCv, YCv, mCv)

    #Desenho da curva de aprendizagem da regressao linear simple
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111)
    ax4.plot(range(1, m+1), error_train, 'o-r', label='Jogo Treinamento')
    ax4.plot(range(1, m+1), error_val, 'x-b', label='Jogo Validacao')
    plt.xlabel('Number of training exemples')
    plt.ylabel('Error')
    plt.title('FIG 4: Linear Regression com polinomio Learning Curve de fator regressao {0}'.format(lbd))
    ax4.legend(loc='best')
    
ValidationCurve(X, Y, m, lbd, XCv, mCv, YCv)

def SelectionLambda(X, Y, m, lbd, XCv, mCv, YCv, listLbd):
    #Se monta um novo jogo de treinamento usando um polinomial de 8 da unica variavel de entrada
    X = PolyFeatures(X[:, 0:], 8, m) #Se desconsidera a constante que esta na 1era coluna
    #Visto a grande diferenca de valor entre as features polinomiais, se aplica a normalizacao
    X, avgX, stdX = FeatureNormalize(X)
    n = X.shape[1] # Novo numero de parametros (1 valor original+8 polinomais deste valor de **2 a **8)
    #Se volta a adicionar a coluna de 1 em 1era posicao para ter a constante da regressao linear
    X = np.hstack((np.ones((m ,1)), X))
    #Se inicializa os parametros de peso theta com 1
    #Valores iniciais dos pesos a passar a funcao de otimizacao
    thetaInit=np.ones((n+1, 1)) #+1 em comparacao ao numero de feature para considerar a constante (=bias)

    #Se monta um novo jogo de Cross Validacao usando um polinomial de 8 da unica variavel de entrada
    XCv = PolyFeatures(XCv[:, 0:], 8, mCv) #Se desconsidera a constante que esta na 1era coluna
    #Visto a grande diferenca de valor entre as features polinomiais, se aplica a normalizacao
    XCv, avgXCv, stdXCv = FeatureNormalize(XCv)
    XCv = np.hstack((np.ones((mCv, 1)), XCv))   #Se adiciona a coluna de 1 em 1era posicao do jogo de Cross Validacao

    error_train = np.array([]) #Vector do Custo J do jogo de treinamento em funcao do tamanho deste jogo
    error_val = np.array([])   #Vector do Custo J do jogo de cross-validacao em funcao do tamanho do jogo de treinamento
    for lbd in listLbd:
        print('Uso do fator de regularizacao ', lbd)
        #Aprendizagem com este fator de regul eo jogo de treinamento completo
        #Treinamento do modelo
        theta, J = TrainLinearReg(thetaInit.ravel(), X, Y, m, n, lbd=lbd)
        print('-> J Treinamento: ', J)
        error_train = np.append(error_train, J) #Info do vector do custo do jogo de treinamento
        #Calculo do custo para o jogo completo de validacao cross usando os param otimizados encima com o jogo parcial de treinamento
        J, gradNaoUsado = LinearRegCostFunction(theta, XCv, YCv, mCv, n, lbd=0) #Nao aplicar regularizacao sobre o jogo de validacao cross
        print('-> J Validacao: ', J)
        error_val = np.append(error_val, J) #Info do vector do custo do jogo de treinamento

    #Se plota os custos J do jogo de dado de treinamento e de o de cross-validacao em funcao do fator de regularizacao
    fig5 = plt.figure(figsize=(10, 6))
    ax5 = fig5.add_subplot(111)
    ax5.plot(listLbd, error_train, 'r', label='Jogo de Treinamento')
    ax5.plot(listLbd, error_val, 'b', label='Jogo de Validacao Cross')
    plt.xlabel('Valor of the Regularizacao fator Lambda')
    plt.ylabel('Error J')
    plt.title('FIG 5: Evo custo com param otim em funcao do fator de regul')
    ax5.legend(loc='best')

    #Selecao do fator de regularizacao que esta o que permite um custo menor no jogo de cross-validacao
    idx = np.argmin(error_val)
    return listLbd[idx] #Retorno deste melhor fator de regularizacao
    
listLbd = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.7, 1, 1.5, 2, 2.5, 3, 10]
lbdOpt = SelectionLambda(X, Y, m, lbd, XCv, mCv, YCv, listLbd)
print('Fator de regularizacao otimo lbdOpt: {0}'.format(lbdOpt))

#Calculo do custo do jogo de teste usando os param otimizados correspondendo ao fator de regul otimo
#Se monta um novo jogo de treinamento usando um polinomial de 8 da unica variavel de entrada
X = PolyFeatures(X[:, 0:], 8, m) #Se desconsidera a constante que esta na 1era coluna
#Visto a grande diferenca de valor entre as features polinomiais, se aplica a normalizacao
X, avgX, stdX = FeatureNormalize(X)
n = X.shape[1] # Novo numero de parametros (1 valor original+8 polinomais deste valor de **2 a **8)
#Se volta a adicionar a coluna de 1 em 1era posicao para ter a constante da regressao linear
X = np.hstack((np.ones((m ,1)), X))
#Se inicializa os parametros de peso theta com 1
#Valores iniciais dos pesos a passar a funcao de otimizacao
thetaInit=np.ones((n+1, 1)) #+1 em comparacao ao numero de feature para considerar a constante (=bias)

#Aprendizagem sobre o jogo de treinamento usando o fator de regul lambda
theta, J = TrainLinearReg(thetaInit.ravel(), X, Y, m, n, lbd=lbdOpt)
print('Custo J Com Regul a {0} com o jogo X treinado com lbd otimo {1}'.format(J, lbdOpt))

#Se monta um novo jogo de Cross Validacao usando um polinomial de 8 da unica variavel de entrada
XTest = PolyFeatures(XTest[:, 0:], 8, mTest) #Se desconsidera a constante que esta na 1era coluna
#Visto a grande diferenca de valor entre as features polinomiais, se aplica a normalizacao
XTest, avgXTest, stdXTest = FeatureNormalize(XTest)
XTest = np.hstack((np.ones((mTest, 1)), XTest))   #Se adiciona a coluna de 1 em 1era posicao do jogo de Cross Validacao

#Calculo do custo J sobre o jogo de test com os parametros theta otimizados sobre o jogo de treinamento com o fato de reg otimo
J, gradNaoUsado = LinearRegCostFunction(theta, X, Y, m, n, lbd=0) #Nao aplicar regularizacao sobre o jogo de validacao cross
print('Custo J Sem Regul a {0} com o jogo X usando os param calculados com lbd otimo {1}'.format(J, lbdOpt))

#Calculo do custo J sobre o jogo de test com os parametros theta otimizados sobre o jogo de treinamento com o fato de reg otimo
J, gradNaoUsado = LinearRegCostFunction(theta, XTest, YTest, mTest, n, lbd=0) #Nao aplicar regularizacao sobre o jogo de validacao cross
print('Custo J Sem Regul a {0} com o jogo XTest usando os param calculados com lbd otimo {1}'.format(J, lbdOpt))

