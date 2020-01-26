# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:23:17 2019

@author: etienne.vanhaecke
SMV com varios jogos de dado de 2D
"""
#%matplotlib #A RODAR UMA VEZ E DEPOIS A COMENTAR PARA OS GRAFICOS APARECER AFORA DA CONSOLE

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.io import loadmat
#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
pasta="C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex6\\ex6\\"

data = loadmat(pasta+"ex6data1.mat")
X = data['X'] #Recuperacao das features de entrada de cada exemplo
Y = data['y'] #Recuperacao do label positivo/negativo de cada exemplo

positivo = X[Y[:, 0]==1] #Exemplos do jogo com label positivo 
negativo = X[Y[:, 0]==0] #Exemplos do jogo com label negativo

#Desenho dos exemplos em funcao das suas duas features
fig, ax = plt.subplots(figsize=(12,8))

ax.scatter(positivo[:, 0], positivo[:, 1], s=50, c='r', marker='o', label='Positivo')
ax.scatter(negativo[:, 0], negativo[:, 1], s=50, c='b', marker='+', label='Negativo')
ax.legend()

X0Min = np.min(X[:, 0]) - 0.2
X0Max = np.max(X[:, 0]) + 0.2

maxIter= 50000

#Treinamento com  um modelo de penalidade (parametro C) baixo para evitar o overfitting (bias maior mas menos variancia)
#O parametro fit_intercept com o valor default a True permite de considerar a constante
svmLinC1 = svm.LinearSVC(C=1, loss = 'hinge', max_iter=maxIter)
svmLinC1.fit(X, Y.ravel())
print(svmLinC1.coef_) #Coeficientes calculados para cada feature (mas nao aparece p coef da constante...)
print(svmLinC1.intercept_) #Coeficiente da constante
print(svmLinC1.score(X, Y)) #Nao esta considerar o outlier
conf = svmLinC1.decision_function(X) #Confianca da predicao em funcao da distancia do ponto da Decision Boundary (mais longe esta melhor a confianca)
#Desenho dos exemplos em funcao das suas duas features e usando uma cor diferente em funcao da confianca calculada encima
#Para os exemplos positivos a confianca deveria estar positiva e para os negativos a confianca deveria estar negativa
XConf = np.hstack((X[:,:], conf[:, np.newaxis])) 
fig2, ax2 = plt.subplots(figsize=(12,8))
ax2.scatter(XConf[:, 0], XConf[:, 1], s=50, c=XConf[:, 2], marker='o', cmap='seismic')
ax2.set_title('SVM (C=1) Decision Confidence')
#Se desenha a linha de boundary decision 
X1_X0Min = (-svmLinC1.intercept_ -svmLinC1.coef_[0,0]*X0Min) / svmLinC1.coef_[0,1]
X1_X0Max = (-svmLinC1.intercept_ -svmLinC1.coef_[0,0]*X0Max) / svmLinC1.coef_[0,1]
ax2.plot([X0Min, X0Max], [X1_X0Min, X1_X0Max])


#Treinamento com  um modelo de penalidade (parametro C) alto para incluir o oulier no modelo (bias zerado sobre o jogo mas variancia maior)
#O parametro fit_intercept a True por default permite de considerar a constante
#O parametro C controla a penalidade aplicada no calculo do custo da funcao de erro
#em caso de mal classificacao. Mais esta alto, mais vai tentar definir um modelo que
#permite de classificar corretamente todos os exemplos do jogo de dados (bias muibo baixo), 
#mas com o risdo de ter uma variancia alta sobre um jogo de dados nao conhecido (problema de overfitting)
#Ele esta similar a 1/lambda que esta o fator de regularizacao dos pesos de modelo de regressao logistica.
svmLinC100 = svm.LinearSVC(C=1000, loss = 'hinge', max_iter=maxIter)
svmLinC100.fit(X, Y.ravel())
print(svmLinC100.coef_) #Coeficientes calculados para cada feature (mas nao aparece p coef da constante...)
print(svmLinC100.intercept_) #Coeficiente da constante
print(svmLinC100.score(X, Y)) #Esta considerado o outlier
conf = svmLinC100.decision_function(X) #Confianca da predicao em funcao da distancia do ponto da Decision Boundary (mais longe esta melhor a confianca)
#Desenho dos exemplos em funcao das suas duas features e usando uma cor diferente em funcao da confianca calculada encima
#Para os exemplos positivos a confianca deveria estar positiva e para os negativos a confianca deveria estar negativa
XConf = np.hstack((X[:,:], conf[:, np.newaxis])) 
fig3, ax3 = plt.subplots(figsize=(12,8))
ax3.scatter(XConf[:, 0], XConf[:, 1], s=50, c=XConf[:, 2], marker='o', cmap='seismic')
ax3.set_title('SVM (C=1000) Decision Confidence')
#Se desenha a linha de boundary decision 
X1_X0Min = (-svmLinC100.intercept_ -svmLinC100.coef_[0,0]*X0Min) / svmLinC100.coef_[0,1]
X1_X0Max = (-svmLinC100.intercept_ -svmLinC100.coef_[0,0]*X0Max) / svmLinC100.coef_[0,1]
ax3.plot([X0Min, X0Max], [X1_X0Min, X1_X0Max])

#Definicao da funcao Gaussian Kernet
def GaussianKernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1-x2) ** 2) / (2 * (sigma ** 2))))
    #Observacao: Nao se pode usar o produto matricial de x1 e -x2 porque
    #precisa ter o quadrado da diferenca dos 2 vectores de cada feature, antes
    #de realizar a soma. Com dot, a soma se faria antes de realizar o quadrado...
    #np.exp(-(np.dot(x1, -1*x2) ** 2) / (2 * (sigma ** 2))) --> Nao funciona     

x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2
GaussianKernel(x1, x2, sigma)

#====================== Jogo de dados 2 =================================
data = loadmat(pasta+"ex6data2.mat")
X = data['X'] #Recuperacao das features de entrada de cada exemplo
Y = data['y'] #Recuperacao do label positivo/negativo de cada exemplo

positivo = X[Y[:, 0]==1] #Exemplos do jogo com label positivo 
negativo = X[Y[:, 0]==0] #Exemplos do jogo com label negativo

#Desenho dos exemplos em funcao das suas duas features
fig4, ax4 = plt.subplots(figsize=(12,8))

ax4.scatter(positivo[:, 0], positivo[:, 1], s=50, c='b', marker='o', label='Positivo')
ax4.scatter(negativo[:, 0], negativo[:, 1], s=50, c='r', marker='+', label='Negativo')
ax4.legend()

#Uso do kernel rbf que esta o default que corresponde ao Gaussian KErnel do curso de valor
#implementado com a funcao GaussianKernel (RBF = Radial Basis Function)
svc = svm.SVC(C=100, gamma=10, probability=True, kernel='rbf')

svc.fit(X, Y.ravel())

conf = svc.decision_function(X) #Confianca da predicao em funcao da distancia do ponto da Decision Boundary (mais longe esta melhor a confianca)
#Desenho dos exemplos em funcao das suas duas features e usando uma cor diferente em funcao da confianca calculada encima
#Para os exemplos positivos a confianca deveria estar positiva e para os negativos a confianca deveria estar negativa
XConf = np.hstack((X[:,:], conf[:, np.newaxis])) 
fig5, ax5 = plt.subplots(figsize=(12,8))
ax5.scatter(XConf[:, 0], XConf[:, 1], s=50, c=XConf[:, 2], marker='o', cmap='seismic')
ax5.set_title('SVM (C=100) Decision Confidence')

prob = svc.predict_proba(X)[:,0]
XProb = np.hstack((X[:,:], prob[:, np.newaxis])) 

fig6, ax6 = plt.subplots(figsize=(12,8))
ax6.scatter(XProb[:, 0], XProb[:, 1], s=50, c=XProb[:, 2], marker='o', cmap='seismic')
ax6.set_title('SVM (C=100) Decision Confidence')
#Exemplo para tracar a decision boundary line
#Passo 1: Recuperar o min e max dos 2 axes do grafico (X0 esta o absicio e x1 no ordenado)
x0Min = X[:, 0].min()
x0Max = X[:, 0].max()
x1Min = X[:, 1].min()
x1Max = X[:, 1].max()
#Passo 2: Uso de uma instancia de mgrid que retorna um grid com 2 arrays 2D mesma shape.
#A 1era com a posicao no axe de absicio (X com X1 no jogo) e a segynda com o ordenado (Y com X2 no jogo)
X0X, X1X = np.mgrid[x0Min:x0Max:200j, x1Min:x1Max:200j]
#Calculo de um valor para cada um dos pontos deste grid que permite de saber em que classe ele esta:
Z = svc.decision_function(np.c_[X0X.ravel(), X1X.ravel()]) #Distancia ate a decision boudary (uso level 0 porque > 0 se positivo senao < 0 se negativo)
Z = svc.predict(np.c_[X0X.ravel(), X1X.ravel()]) #Predicao da classe com 1 ou 0 (uso level a 0.5 ja que positivo se 1 senao 0)
Z = svc.predict_proba(np.c_[X0X.ravel(), X1X.ravel()])[:,0] #Predicao da classe com 1 ou 0 (uso level a 0.5 ja que positivo se 1 senao 0)
Z = Z.reshape(X0X.shape)
#ax6.pcolormesh(X0X, X1X, Z > 0, cmap=plt.cm.Paired)
#Passo 3: Se desenha a boundary line com o grid e o valor que separa as classes, indicando o valor medio (level)
#Poissivel de indicar varias seeparacoes usando diferentes levels, colors e linestyle
ax6.contour(X0X, X1X, Z, colors=['k'],
                linestyles=['-'], levels=[0.5])

print(svc.score(X, Y))

#Serializacao do modelo para conservar ele em disco e reusar ele depois
#para realizar novas predicoes
filename = pasta+'finalized_model.sav'
#Solucao 1
import pickle
# save the model to disk
pickle.dump(svc, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
#Uso do modelo recuperado do arquivo
result = loaded_model.score(X, Y)
print(result)

#Solucao 2
import joblib
# save the model to disk
joblib.dump(svc, filename)
# load the model from disk
loaded_model = joblib.load(filename)
#Uso do modelo recuperado do arquivo
result = loaded_model.score(X, Y)
print(result)

#====================== Jogo de dados 3 =================================
data = loadmat(pasta+"ex6data3.mat")
# Preparacao e visualizacao do jogo de dados de treinamento
X = data['X'] #Recuperacao das features de entrada de cada exemplo do jogo de dados de treinamento
Y = data['y'] #Recuperacao do label positivo/negativo de cada exemplo do jogo de dados de treinamento
positivo = X[Y[:, 0]==1] #Exemplos do jogo com label positivo 
negativo = X[Y[:, 0]==0] #Exemplos do jogo com label negativo
#Desenho dos exemplos em funcao das suas duas features
fig7, ax7 = plt.subplots(figsize=(12,8))
ax7.scatter(positivo[:, 0], positivo[:, 1], s=50, c='b', marker='o', label='Positivo')
ax7.scatter(negativo[:, 0], negativo[:, 1], s=50, c='r', marker='+', label='Negativo')
ax7.legend()
ax7.set_title("Jogo de dados de treinamento")

# Preparacao e visualizacao do jogo de dados de cross-validacao
XVal = data['Xval'] #Recuperacao das features de entrada de cada exemplo do jogo de dados de cross-validacao
YVal = data['yval'] #Recuperacao do label positivo/negativo de cada exemplo do jogo de dados de cross-validacao
positivo = XVal[YVal[:, 0]==1] #Exemplos do jogo com label positivo 
negativo = XVal[YVal[:, 0]==0] #Exemplos do jogo com label negativo
#Desenho dos exemplos em funcao das suas duas features
fig8, ax8 = plt.subplots(figsize=(12,8))
ax8.scatter(positivo[:, 0], positivo[:, 1], s=50, c='b', marker='o', label='Positivo')
ax8.scatter(negativo[:, 0], negativo[:, 1], s=50, c='r', marker='+', label='Negativo')
ax8.legend()
ax8.set_title("Jogo de dados de cross-validacao")

#Diferentes valores dos parametros do modelo SMV a verificar
#Para cada combinacao se treina o modelo com os valores da combinacao e o jogo de treinamento
# e se verifica o resultado sobre o jogo de cross-validacao.
#Se vai conservar os parametros de melhor acuracidade no jogo de validacao cross
CValues = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
gammaValues = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
bestResult = 0
bestParam = {"C": None, "Gamma": None}
for CCor in CValues:
    for gammaCor in gammaValues:
        modelo = svm.SVC(C=CCor, gamma=gammaCor, probability=True, kernel='rbf')
        modelo.fit(X, Y.ravel())
        
        result = modelo.score(XVal, YVal)
        print("-- O modelo SVM com C {0} e gamma {1} da um resultado {2}"
              .format(CCor, gammaCor, result))
        
        if result > bestResult:
            print("---> Melhor resultado ate agora") 
            bestResult = result
            bestParam['C'] = CCor
            bestParam['Gamma'] = gammaCor
print("Melhor resultado {0} com o valor C {1} e gamma {2}"
      .format(bestResult, bestParam['C'], bestParam['Gamma']))

#Montagem do melhor modelo
bestModelo = svm.SVC(C=bestParam['C'], gamma=bestParam['Gamma'], probability=True, kernel='rbf')
bestModelo.fit(X, Y.ravel())
print('Result com o jogo de dados de treinamento '+str(bestModelo.score(X, Y)))
print('Result com o jogo de dados de cross-validacao '+str(bestModelo.score(XVal, YVal)))


#Se desenha a decision boudary line nos 2 graficos do jogo de dado de treinamento e de cross-validacao
x0Min = X[:, 0].min()-0.1
x0Max = X[:, 0].max()+0.1
x1Min = X[:, 1].min()-0.1
x1Max = X[:, 1].max()+0.1
X0X, X1X = np.mgrid[x0Min:x0Max:200j, x1Min:x1Max:200j]
Z = bestModelo.predict_proba(np.c_[X0X.ravel(), X1X.ravel()])[:,0] #Predicao da classe com 1 ou 0 (uso level a 0.5 ja que positivo se 1 senao 0)
Z = Z.reshape(X0X.shape)
ax7.contour(X0X, X1X, Z, colors=['k'], linestyles=['-'], levels=[0.5])
ax8.contour(X0X, X1X, Z, colors=['k'], linestyles=['-'], levels=[0.5])
            
            