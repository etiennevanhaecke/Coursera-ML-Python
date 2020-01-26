# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:02:25 2019

@author: e_van
Exercicio 2.1 do curso de ML de Coursera: Regressao logistica para classificar aceite estudantes universidade em funcao notas 
"""

import numpy as np
import matplotlib.pyplot as plt

#Importacao dos dados do arquivo TXT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
data = np.loadtxt("C:\\Users\\e_van\\IA\\Coursera ML\\machine-learning-ex2\\ex2\\ex2data1.txt", delimiter=',')
m = len(data) #Tamanho do jogo de dados

Y=data[:, -1].reshape(m, 1) #Conservacao do label (classificacao real se o estudante foi ou nao aceoto) do jogo de dados

X=data[:, :-1] #Recuperacao das variaveis de entrada do jogo de dados ate ante ultima coluna
n = X.shape[1] #Numero de features recuperados do numero de coluna do shape do X

#Se vai desenhar o jogo de dados, usando uma cor diferente por classificacao diferente (aceito ou nao na universidade) em funcao das 2 notas
aceito=X[(Y==1)[:, 0], 0:] #Recuperacao das notas do grupo de estudante aceito
recusado=X[(Y==0)[:, 0], 0:] #Recuperacao das notas do grupo de estudante recusado
#OBS: Outra maneira de montar estes 2 grupos seria via list comprehension
plt.close('all') 
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.scatter(x=aceito[:, 0], y=aceito[:, 1], c='b', label='Aceito') #Desenho dos estudantes aceitos em Azul
ax1.scatter(x=recusado[:, 0], y=recusado[:, 1], c='r', label='Recusado') #Desenho dos estudantes recusados em Vermelho
plt.legend(loc='best') #Opcao best nao implementada usando legend com plt
plt.xlabel('Nota Examem 1')
plt.ylabel('Nota Examem 2')
ax1.set_title('Classificacao em funcao notas')
plt.show()

#Funcao de normalizacao a aplicar a todos os jogos de dado multi-variaveis que tenha ou nao diferenca de escala entre as variaveis
def featureNormalization(X):
    meanX = np.mean(X, axis=0) #Media de cada feature (=coluna do jogo de dados)
    stdX = np.std(X, axis=0) #Desvio padrao de cada feature (=coluna do jogo de dados)
    X = (X-meanX) / stdX  #Aplicacao da normalizacao
    return X, meanX, stdX

X, meanX, stdX = featureNormalization(X)

#Como sempre, para considerar o bias no modelo, se adiciona uma 2era coluna de 1 para o parametro constante
X=np.concatenate((np.ones((m, 1)), X), axis=1)

#Parametros das variaveis de entrada inicializada por default a 0 (considerando o param da constante)
theta = np.zeros((n+1, 1))
initial_theta= theta.ravel().copy() #Usado na funcao de otimizacao e deve estar um vector linear de 1 dimensao (n,)
                                    #IMPORTANTE: Pensar a usar copy senao seeria so uma nova referencia sobre o mesmo espaco memoria que theta
def Sigmoid(z):
    z = np.array(z) #Caso z seja passado como lista
    return 1 / (1 + np.exp(-z))

def CalculoCusto(theta, X, Y, m, n):
    theta = theta.reshape((n+1), 1) #Se redimensiona ja que para estar usado na proc de otimizacao, tem que passar theta como vector linear de 1 dimensao (n,)
    H=Sigmoid(np.dot(X, theta)) #Calculo da predicao de classificacao de cada estudante do jogo de dados em funcao das notas X e da parametrizacao theta
    J1 = 1/m * np.sum(( - Y*np.log(H)) - (1-Y)*(np.log(1-H)))
    #Abaixo outra maneira de calcular trocando o uso da funcao soma pela multiplicacao matricial (TALVE MAIS RAPIDO)
    J2 = 1/m * ( - np.dot(Y.T, np.log(H)) - np.dot((1-Y).T, np.log(1-H)))
    return J2.ravel() #Se fica com uma unida dimensao para nao ter problema durante a historizacao deste custo na funcao GradientDescent
    
#Test da funcao CalculoCusto com os valores iniciais de theta a 0
numIter = 0
J = CalculoCusto(theta, X, Y, m, n)
print("Custo inicial {0} com os param theta inicializados a 0".format(J))

alpha = 1 #Fator de aprendizagem
limite = 0.000001 #Variacao em valor absoluto de baixa da funcao custo J entre 2 iteracoes do gradient descent para finali\ar este gradient

#Calculo do derivado da funcao custo (theta deve estar em 1era posicao para pode estar chamada pela funcao de otimizacao)
def Gradient(theta, X, Y,  m, n):
    theta = theta.reshape((n+1), 1) #Se redimensiona ja que para estar usado na proc de otimizacao, tem que passar theta como vector linear de 1 dimensao (n,)
    H = Sigmoid(np.dot(X, theta)) #Calculo da previsao
    erro = np.dot(X.T, (H-Y)) #Equivalente de somar o erro de todas as linhas do jogo de dado
    gradient = 1 / m * erro
    return gradient

#Definidao da funcao de gradient descent que vai otimizar passo a passo os parametros theta
def GradientDescent(X, Y, theta, alpha, limite, J, n, m):
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
        gradient = Gradient(theta, X, Y, m, n)
        print('-> Gradient {0} a aplicar nos parametros theta.'.format(gradient))
        theta -= alpha*gradient
        
        J = CalculoCusto(theta, X, Y, m, n) #Calculo do custo com os novos parametros theta
        histJ = np.vstack((histJ, np.hstack((theta.T.ravel(), J)))) #Historisacao do custo por iteracao
                
        if np.abs(J-histJ[-2, -1]) < limite:
            break; #Saida da descida do gradient ja que ela encontrou um pomnto baixo
    
    return(numIter, J, theta, histJ)
    
numIter, J, theta, histJ = GradientDescent(X, Y, theta, alpha, limite, J, n, m)
print("Gradient descent finalizado depois de {0} iteracoes: Custo J esta {1} e os theta {2}".format(numIter, J, theta.ravel()))

#Se desenha o grafico do custo J em funcao do numero de iteracao do gradient descendente
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.scatter(range(numIter+1), histJ[:, -1], c='y', marker='.', label = 'Custo J por Iteracao')
plt.legend(loc='best') #Opcao best nao implementada usando legend com plt
plt.xlabel('Num. Iteracao')
plt.ylabel('Custo J')
ax1.set_title('Custo J em funcao iteracao')
plt.show()

#Uso de funcoes de optimizacao da biblioteca scipy.optimize
from scipy.optimize import minimize, fmin_tnc, fmin, fmin_bfgs, fmin_ncg, leastsq, fmin_slsqp

Result = minimize(fun = CalculoCusto, x0 = initial_theta, args = (X, Y, m, n),
                  method = 'TNC', jac = Gradient);
optTheta = Result.x;
optJ = Result.fun
print('Com minimize de scipy.optimize se chega a um custo optJ de {0} e optTheta {1}'.format(optJ, optTheta))

Result = fmin_tnc(func=CalculoCusto, x0=initial_theta, args=(X, Y, m, n), fprime=Gradient)
tncTheta = Result[0];
print('Com fmin_tnc de scipy.optimize se chega a tncTheta {0}'.format(tncTheta))

#Versao sem passar a funcao de calculo do gradient (parametro fprime), informando o param approx_gradbool com True
Result = fmin_tnc(func=CalculoCusto, x0=initial_theta, args=(X, Y, m, n), approx_grad=True) 
tncTheta = Result[0];
print('Com fmin_tnc de scipy.optimize, SEM PASSAR A FCT GRADIENT, se chega a tncTheta {0}'.format(tncTheta))

#Classificaco de um novo estudante de nota1 45 e nota2 85
#Primeiro tem que pensar a normalizar estas notas, ja que os parametros otimizados foram calculados com o jogo 
#de dado X normalizado
notas=np.array([45, 85])
notas=(notas-meanX)/stdX
#Segundo tem que pensar a adicionar a constante 1 no inicio do vector
notas=np.hstack(([1], notas))
#Terceiro se aplica a formula H de deducao da classificacao
Hnotas = Sigmoid(np.dot(notas, tncTheta))
print('A previsao de admissibilidade esta de {0}'.format(round(Hnotas*100, 2)))

#A linha de separacao da hipotese esta definida com H = 0.5. Com esta definicao, se consegue a definir a
#nota 2 que corresponde a cada nota 1 para ficar com H a 0.5 e a tracar a linha (decision boundary)
#Se desenha ela na figura 1 onde ja foram desenhados os diferentes alunos do jogo de dados em funcao das notas deles
xDB= np.array([np.min(X[:,1]),np.max(X[:,1])]) #Intervalo entre o min e max da nota 1 que esta no axe X absice e em segunda posicao no array X
yDB=-(theta[0] +theta[1]*xDB)/theta[2] #Se calcula a nota 2 que corresponde a nota 1 para H a 0,5, que servira para o ordenado do axe Y
xDB = xDB*stdX+meanX #Se denormaliza xDB, ja que ele foi calculado com a nota 1 de X normalizado para voltar a ter a nota 1 normal do absice x da linha Decision Boudary
yDB = yDB*stdX+meanX #Se denormaliza yDB ja que pelo uso dos param theta obtidos com X normalizados, ele corresponde a uma nota 2 normalizada, para voltar a ter a nota 2 normal do ordenado y da linha Decision Boudary
ax1.plot(xDB,yDB, "g") #Desenho da linha de Decision Boundary no grafico 1 em verde

#Funcao para aplicar a predicao sobre o jogo de dados X considerado ja normalizado e contendo a 1era coluna de 1 para i bias/constante
def Predict(X, theta, n):
    theta = theta.reshape((n+1), 1) #Se garante que theta esta um vector vertical (=matrice de 1 coluna) 
                                    #para poder realizar o produto matricial com X de numero coluna (com cst) = num linha de theta 
    #print(theta.shape)
    H = np.round(Sigmoid(np.dot(X, theta)), 2)
    return H

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

H = Predict(X, theta, n)   #Predicao usando os parametros theta otidos com o gradient descent
H = np.round(H)            #Se arredonda para comparar contra a classificacao real
compHcomY = np.hstack((H, Y)) #Comparacao contra a classificacao real colocando o previsto e o label em um mesmo array
confMatrix = confusion_matrix(Y, H) #Confusion matrix apresentando em uma matrice de 2X2, com label em linha e predicao em coluna,
                                    #o RN (Right Negative), o FP (False Positive) na PRIMEIRA linha E
                                    #o FN (False Negative), o RP (Rigt Positive) na SEGUNDA linha
accScore = accuracy_score(Y, H) #ACCURACIDADE: Fracao de exemplos classificados corretamente = (TN+TP) / (TN+FP+FN+TP)
recScore = recall_score(Y, H) #RACALL: Fracao de exemplos positivos (aceitos na universidade) corretamente classificados = RP / (RP+FN)
precScore = precision_score(Y, H) #PRECISION: Fracao de exemplos predictos como positivos que realmente estao positivos: = RP / (RP+FP)
F1Score = f1_score(Y, H) #f1 score is the harmonic mean of recall and precision, with a higher score as a better model
#Tentar Grafico mostra os 4 casos TN+FP+FN+TP, usando a forma para indicar o label (o para aceito e + para recusado) E
#a cor para indicar a predicao (verde aceito e vermelho recusado). 
#Assim +Vermelho = TN, +Verde = FP, oVermelho = FN e oVerde = TP
HAceito = np.round(Predict(np.hstack( (np.ones((len(aceito),1)),(aceito-meanX)/stdX) ), theta, n))
TP = aceito[HAceito.ravel()==1] #Grupo dos exemplos True Positive (de label aceito e de previsao aceito)
FN = aceito[HAceito.ravel()==0] #Grupo dos exemplos False Negative (de label aceito e de previsao recusado)
HRecusado = np.round(Predict(np.hstack( (np.ones((len(recusado),1)),(recusado-meanX)/stdX) ), theta, n))
TN = recusado[HRecusado.ravel()==0] #Grupo dos exemplos True Negative (de label recusado e de previsao recusado)
FP = recusado[HRecusado.ravel()==1] #Grupo dos exemplos False Positive (de label recusado e de previsao aceito)
plt.close()
fig3=plt.figure(figsize=(10,6)) #Niva figura 3
ax3=fig3.add_subplot(1,1,1) #Se desenha o axe neste grafico
ax3.scatter(x=TP[:, 0], y=TP[:, 1], c='green', marker='o', label='TP - True Positive')
ax3.scatter(x=FN[:, 0], y=FN[:, 1], c='red', marker='o', label='FN - False Negative')
ax3.scatter(x=TN[:, 0], y=TN[:, 1], c='red', marker='+', label='TN - Tue Negative')
ax3.scatter(x=FP[:, 0], y=FP[:, 1], c='green', marker='+', label='FP - False Positive')
plt.legend(loc='best') #Opcao best nao implementada usando legend com plt
plt.xlabel('Nota Examem 1')
plt.ylabel('Nota Examem 2')
ax3.set_title('Os 4 grupos de classificaca: TP - FN - TN -FP')
plt.show()

H = Predict(X, optTheta, n)   #Predicao usando os parametros theta otidos com a funcao de otimizacao de scipy
H = np.round(H)            #Se arredonda para comparar contra a classificacao real
compHcomY = np.hstack((H, Y)) #Comparacao contra a classificacao real colocando o previsto e o label em um mesmo array

