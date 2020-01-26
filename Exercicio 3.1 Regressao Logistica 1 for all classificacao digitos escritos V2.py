# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:21:31 2019

@author: e_van
Exercicio 3.1 do curso de ML de Coursera: Regressao linear one-vs-all para identificacao dos digitos escritps manualmente
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
data = loadmat("C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex3\\ex3\\ex3data1.mat", variable_names=('X', 'y'))

from sklearn.model_selection import train_test_split

X=data['X']
n=X.shape[1] #Numero de feature (intensidade de gris de cada pixel da imagem do digito)

Y=data['y'] #Recuperacao do label (digito de 0 a 9), com 0 representado por 10
Y[Y[:, 0]==10, 0] = 0 #Se transforma o valor 10 representando o digito 0 por 0

#Para verificar que nao tem uma grande variance (overfit do modelo sobre o jogo de treinamento), se divide
#o jogo em 2: 75% selecionados aleatoriamente para o treinamento (mantendo os nomes X e Y para nao mudar o codigo)
#e 25% para o teste da acuracidade sobre um jogo que nao foi usado no treinamento
X, XTest, Y, YTest = train_test_split(X, Y, test_size=0.25, random_state=1)

m = len(X) #Tamanho do jogo de dados de treinamemto
mTest = len(XTest) #Tamanho do jogo de dados de test

#Se vai desenhar em uma figura com um grid de 10 sobre 10, 100 digitos selecionados de maneira aleatoria
plt.close('all') #Fechamento de todas as figuras abertas
from random import seed, sample
seed() #Inicializa o generator de numero aleatorio a parte da hora do sistema
numSampleX = 6 #Numero de linha do sample para apresentacao aleatoria
numSampleY = 17 #Numero de coluna do sample para apresentacao aleatoria
idxRdm = sample(range(m), (numSampleX*numSampleY)) #Selecao aleatoria de 100 index dentro do range dos 5000 exemplos do jogo de dados
fig1 = plt.figure(figsize=(26, 12))       #Criacao da figura 1 onde apresentar a mostragem de 100 fotos de digito selecionados aleatoriamente
plt.xticks(()) #Para retirar o axe X
plt.yticks(()) #Para retirar o aze Y
#Se apresenta o titulo, deixando explicito o dicionario fontdict dos atributos do print deste titulo (opcional). Entender como ajuste esta config 
#para ter um titulo maior. Por exemplo plt.rcParams['axes.titlesize'] = 'large'
plt.title('Apresentacao aleatoria de {0} fotos de digitos'.format((numSampleX*numSampleY)),
          fontdict={'fontsize': plt.rcParams['axes.titlesize'], 'fontweight' : plt.rcParams['axes.titleweight'],
                    'verticalalignment': 'baseline', 'horizontalalignment': 'center'})
for i in range((numSampleX*numSampleY)):   #Para cada um das fotos da mostragem
    ax = fig1.add_subplot(numSampleX, numSampleY, (i+1)) #Selecao do subplot onde apresenta a foto, fazendo cuidado ao numero do subplot que comeca a 1 e nao a 0 (por isso i+1)
    plt.xticks(()) #Para retirar o axe X
    plt.yticks(()) #Para retirar o aze Y
    #Se apresenta a foto do digito passando as 100 features de intensidade de cor nas dimensoes 20X20 da foto, ordenando no formato Fortran (1ero anda na coluna antes linha) para 
    #a foto estar orientada corretamente. Se indica a mapa de cor de declinacao de gris com o parametro cmap.
    ax.imshow(X[idxRdm[i]].reshape(20, 20, order = 'F'), cmap='Greys') 
    ax.set_title('label {0}'.format(Y[idxRdm[i], 0]))
    
plt.tight_layout() #Ajusta a dimensao do window a dimensao dos sub-plots
plt.show()

#Se vai montar um classificador para cada digito, com os parametros theta de cada classificador definidos por regressao logistica regularizada
#Cada classificador vai indicar um % de probabilidade da foto do digito corresponder ao digido do classificador.
#Para isso se deve definir um label por classificador indicando sim (valor 1) ou nao (valor 0) o digito corresponde ao digito da classe do classificador

X=np.hstack(( np.ones((m, 1)), X) ) #Se adiciona ao array de feature do jogo de treinamento a coluna de 1 da constante
XTest=np.hstack(( np.ones((mTest, 1)), XTest) ) #Se adiciona ao array de feature do jogo de treinamento a coluna de 1 da constante

YClas=np.zeros((m, 10)) #Inicializaao do array de label dos 10 classificadores
for i in range(10): #Para cada digito de 0 a 9 presente no label Y
    #Para cada index de exemplo digito cujo o valor corresponde ao digito do loop Y[:, 0]==i
    #se posiciona 1 (SIM) na coluna do array que corresponde a classe do classificador deste digito , i] = 1 
    YClas[Y[:, 0]==i, i] = 1 

thetaClas=np.zeros(((n+1), 10)) #Inicializaao do array dos parametros theta da constante e das diferentes features (400) de cada um dos 10 classificador (um por digito de 0 a 9)
initial_thetaClas= thetaClas.copy() #Usado na funcao de otimizacao e deve estar um vector linear de 1 dimensao (n,)
                                    #IMPORTANTE: Pensar a usar copy senao seeria so uma nova referencia sobre o mesmo espaco memoria que theta

def Sigmoid(z):
    z = np.array(z) #Caso z seja passado como lista
    return 1 / (1 + np.exp(-z))

def CalculoCustoRegClas(thetaClas, X, YClas, m, n, lbd):
    thetaClas = thetaClas.reshape((n+1), 10) #Se redimensiona como array de 10 colunas (uma por classe a classificar)
    HClas=Sigmoid(np.dot(X, thetaClas)) #Calculo da probabilidade das 10 predicao de classificacao de cada digito do jogo de dados em funcao das intensidades de gris X e da parametrizacao theta de cada classe digito
    #Calculo do custo de cada um dos 10 classificadores fazendo uma soma do desvio de cada exemplo digito (os 5000 de X) 
    J1Clas = 1/m * np.sum(( - YClas*np.log(HClas)) - (1-YClas)*(np.log(1-HClas)), axis=0) #Por soma numpy por coluna
    J2Clas = np.diagonal(1/m * ( - np.dot(YClas.T, np.log(HClas)) - np.dot((1-YClas).T, np.log(1-HClas)))) #Pela multiplicacao matricial (TALVE MAIS RAPIDO) recuperando so a diagonal que corresponde ao dot da linha e de coluna de mesmo index
    #np.einsum('ij,ji->i', a, b) #Solucao possivel para o calculo so da diagonal (dot de linha com coluna de mesmo index) sem realizar todo o calculo matricial dos 2 arrays
    #Regularizacao com lambda usando o metodo Ridge
    JRegClas = J2Clas + (lbd/(2*m) * np.sum(thetaClas**2, axis=0)) #CUIDADO a indicar em que axe realizar a soma, senao np.sum soma todos os elementos sobre os diferentes axes  
    return JRegClas #Retorno de um vector com o custo de cada classificador (um por digito de 0 a 9)

lbd = 1 #Fator de regularizacao para evitar o overfitting penalisando as hipoteses com maior soma de parametros theta
alpha = 0.7 #Fator de aprendizagem
limite = 0.000001 #Variacao em valor absoluto de baixa da funcao custo J entre 2 iteracoes do gradient descent para finali\ar este gradient
    
#Test da funcao CalculoCustoRegClas com os valores iniciais de theta a 0
numIter = 0
JClas = CalculoCustoRegClas(thetaClas, X, YClas, m, n, lbd) #Custo inicial de cada classificador com os param theta de cada classificador a 0
print("Custo inicial {0} de cada classificador (digito 0 a 9) com os param theta de cada classe inicializados a 0".format(JClas))

#Calculo do derivado da funcao custo (theta deve estar em 1era posicao para pode estar chamada pela funcao de otimizacao)
def GradientRegClas(thetaClas, X, YClas,  m, n, lbd):
    thetaClas = thetaClas.reshape((n+1), 10) #Se redimensiona com array de 10 colunas (uma por classificador de classe de digito de 0 a 9)
    HClas = Sigmoid(np.dot(X, thetaClas)) #Calculo da probabilidade das 10 predicao de classificacao de cada digito do jogo de dados em funcao das intensidades de gris X e da parametrizacao theta de cada classe digito
    erroClas = np.dot(X.T, (HClas-YClas)) #Equivalente de somar o erro de todas as linhas do jogo de dado
    #Se regulariza com o metodo Ridge so o gradient dos parametros theta de feature (a constante/bias nao esta a regularizar)
    erroClas[1:] = erroClas[1:] + lbd/m*thetaClas[1:]
    gradientClas = 1 / m * erroClas
    return gradientClas

#Definidao da funcao de gradient descent que vai otimizar passo a passo os parametros theta
def GradientDescentClas(X, YClas, thetaClas, alpha, limite, JClas, n, m, lbd):
    numIter = 0
    #Se inicializa o coficiente de aprendizagem de cada classificador/hipotese com o param alpha
    alphaClas = np.array([alpha]*10) 
    #1era Historizacao do custo J por it eracao
    histJClas = np.empty((1, 10)) #Para historizar o custo de cada um dos 10 classificadores (1 por digito) por iteracao
    histJClas[0] = JClas[:] #Linha 0 do array do historizacao com o custo inicial de cada classificador/hipotese
        
    while True:
        print("Iteracao {0}: Custo JClas esta {1}".format(numIter, JClas))
        numIter += 1 #Incremento do numero de iteracao
        
        #Ajuste dos prametros em funcao do derivado da funcao custo
        gradientClas = GradientRegClas(thetaClas, X, YClas, m, n, lbd) #Calculo do gradient por variavel de entrada (considerando a constante) e por hipotese/classificador
        #print('-> Gradient {0} a aplicar nos parametros theta.'.format(gradient))
        thetaClas -= alphaClas*gradientClas
        
        JClas = CalculoCustoRegClas(thetaClas, X, YClas, m, n, lbd) #Calculo do custo com os novos parametros theta
        histJClas = np.vstack((histJClas, JClas)) #Historisacao do custo de cada classificador para a nova iteracao
        
        #Se cada classificador/hipotese tem uma variacao de custo (custo da iteracao - custo da iteracao anterior)
        #inferior ao limite passado em parametro
        #print(np.abs(JClas-histJClas[-2]))
        #print(np.abs(JClas-histJClas[-2])<limite)
        #print(alphaClas)
        #print(thetaClas[0::200])
        if np.all(np.abs(JClas-histJClas[-2])<limite):
            break; #Saida da descida do gradient ja que todas as hipoteses/classificadores se encontraram no ponto baixo
    
        #Se zera o coeficiente de aprendizagem alpha das classes/classificadores/hipoteses que ja chegaram ao ponto baixo
        #para nao seguir o gradient descent (se congela os param theta deles ao valor do ponto baixo) para estes classificadores 
        #(senao tem o risco deles desviar e o custo deles aumentar). So para os elementos de alphaClas de index com a condicao
        #np.abs(JClas-histJClas[-2]) < limite a true que se forca 0.
        #alphaClas[np.abs(JClas-histJClas[-2])<limite] = 0 --> FACULTATIVO PORQUE FOI UMA CONTINGENCIA DEVIDO AO BUG DA REGULARIZACAO
                                                             # DO CUSTO JClas por classificador feita erroneamente somando o quadrado 
                                                             # dos theta de cada feature+constante para TODAS AS HIPOTESES.
                                                             # O correto esta de regulaorizar o custo de cada classificador considerando
                                                             # os theta de cada classificador.
                                                             # CORRIGIDO indicando o axe vertical (axis=0) como parametro da funcao np,sum
                                                             
        
    return(numIter, JClas, thetaClas, histJClas)
    
numIter, JClas, thetaClas, histJClas = GradientDescentClas(X, YClas, thetaClas, alpha, limite, JClas, n, m, lbd)
print("Gradient descent finalizado depois de {0} iteracoes: Custo J esta {1}".format(numIter, JClas))

#Se desenha o grafico do custo JClas de cada classificador em funcao do numero de iteracao do gradient descendente
#plt.close()
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(1, 1, 1)
for i in range(10): #para cada classificador de digito de 0 a 9
    ax2.scatter(range(numIter+1), histJClas[:, i], marker='.', label = 'Evo por iteracao do custo J do classificador do digito {0}'.format(i))
plt.legend(loc='best') #Opcao best nao implementada usando legend com plt
plt.xlabel('Num. Iteracao')
plt.ylabel('Custo J')
ax2.set_title('Custo J de cada classificador digito 0 a 9 em funcao numero iteracao')
plt.show()

#Teste de deducao do digito da 1era e ultima foto do jogo de dados, aplicando os 10 classificadores/hipoteses
#sobre as 400 features de intensidade de gris (+constante em 1era posicao) destas fotos e selecionando o digito
#de probabilidade da hipotese maior.
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
plt.imshow(X[0, 1:].reshape(20, 20, order='F'), cmap='Greys') 
prob0a9 = Sigmoid(np.dot(X[0, :], thetaClas))
probMax = np.amax(prob0a9) #Caso precisa do max por coluna ou linha em array de 2D, usar o param axis = 0 por coluna e axis=1 por linha)
digProbMax = np.where(prob0a9 == probMax)[0] #O digito corresponde ao index da probabilidade maior
print('Probabilidade maior {0} correspondendo ao digito {1} das probabilidades {2} dos digitos de 0 a 9.'.format(probMax, digProbMax, prob0a9))
plt.title('Label {0} com deducao {1} de probab {2}%'.format(Y[0], digProbMax, round(probMax*100, 2)))

plt.imshow(X[-1, 1:].reshape(20, 20, order='F'), cmap='Greys')
prob0a9 = Sigmoid(np.dot(X[-1, :], thetaClas))
probMax = np.amax(prob0a9) #Caso precisa do max por coluna ou linha em array de 2D, usar o param axis = 0 por coluna e axis=1 por linha)
digProbMax = np.where(prob0a9 == probMax)[0] #O digito corresponde ao index da probabilidade maior
print('Probabilidade maior {0} correspondendo ao digito {1} das probabilidades {2} dos digitos de 0 a 9.'.format(probMax, digProbMax, prob0a9))
plt.title('Label {0} com deducao {1} de probab {2}%'.format(Y[-1], digProbMax, round(probMax*100, 2)))

#Loop foto a foto sobre a mostragem de foto de digito apresentada na figura 1
'''for i in range((numSampleX*numSampleY)):   #Para cada um das fotos da mostragem
    plt.imshow(X[idxRdm[i], 1:].reshape(20, 20, order='F'), cmap='Greys')
    prob0a9 = Sigmoid(np.dot(X[idxRdm[i], :], thetaClas))
    probMax = np.amax(prob0a9) #Caso precisa do max por coluna ou linha em array de 2D, usar o param axis = 0 por coluna e axis=1 por linha)
    digProbMax = np.where(prob0a9 == probMax)[0] #O digito corresponde ao index da probabilidade maior
    print('Idx Img {0}: Probabilidade maior {1} correspondendo ao digito {2} das probabilidades {3} dos digitos de 0 a 9.'.format(idxRdm[i], probMax, digProbMax, prob0a9))
    plt.title('Idx {0}: Label {1} com deducao {2} de probab {3}%'.format(idxRdm[i], Y[idxRdm[i]], digProbMax, round(probMax*100, 2)))
    plt.pause(1) #Espera 1 segundo antes de passar a imagem seguinte
'''
#Previsao para cada elemento (linha) de X do digito de 0 a 9 a parte das features da foto (+constante 1 era posicao) e dos parametros theta de cada classificador/hipotese dos digitos 0 a 9
def Previsao(X, thetaClas, m, n):
    thetaClas = thetaClas.reshape((n+1), 10) #Caso nao esta passado como vector de coluna (matrice de 1 coluna)
    Prob = Sigmoid(np.dot(X, thetaClas)) #Calculo da probabilidade de cada um dos 10 classificadores de 0 a 9
    ProbMax = np.amax(Prob, axis=1) #Selecao da hipotese maior por linha exemplo 
    return np.where(Prob == ProbMax.reshape(m, 1))[1]     #Se retorna para cada linha exemplo de X o digito do classificador de probabilidade maior (caso tem varios de mesma probabilidade maior se retorna o 1ero da lista de numero crescente)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

#Verificacao da acuracidade dos preditores sobre o jogo de dados de treinamento
print('Acuracidade dos 10 preditores sobre o jogo de dados de treinamento de {0} exemplos '.format(m))
H=Previsao(X, thetaClas, m, n) #Digito previsto para cada linha exemplo de X
compHcomY = np.hstack((H.reshape(m, 1), Y)) #Comparacao contra a classificacao real colocando o previsto e o label em um mesmo array
for i in range(10):
    #print(i)
    tot = len(Y[Y==i])
    #print(tot)
    ok = np.sum(H.reshape(m, 1)[Y==i]==Y[Y==i])
    #print(ok)
    print('Digito {0}: {1} classificados corretamente sobre os {2} do jogo - {3}%'
          .format(i, ok, tot, np.round(ok/tot*100, 2)))

#Verificacao da acuracidade dos preditores sobre o jogo de dados de test
print('Acuracidade dos 10 preditores sobre o jogo de dados de test de {0} exemplos '.format(mTest))
HTest=Previsao(XTest, thetaClas, mTest, n) #Digito previsto para cada linha exemplo de X
compHTestcomYTest = np.hstack((HTest.reshape(mTest, 1), YTest)) #Comparacao contra a classificacao real colocando o previsto e o label em um mesmo array
for i in range(10):
    #print(i)
    tot = len(YTest[YTest==i])
    #print(tot)
    ok = np.sum(HTest.reshape(mTest, 1)[YTest==i]==YTest[YTest==i])
    #print(ok)
    print('Digito {0}: {1} classificados corretamente sobre os {2} do jogo - {3}%'
          .format(i, ok, tot, np.round(ok/tot*100, 2)))

#Confusion matrice sobre o jogo de teste
confusion_matrix(YTest, HTest)
accuracy_score(YTest, HTest)
recall_score(YTest, HTest, average = 'weighted')
precision_score(YTest, HTest, average = 'macro')
f1_score(YTest, HTest, average = 'micro')

#Uso de funcoes de optimizacao da biblioteca scipy.optimize
from scipy.optimize import minimize

#Para isso precisa ter a funcao de calculo de custo e de gradient para uma classificacao binaria
#(as encima foram preparadas para ter os 10 classificadores calculados de uma vez por calculo matrocial)

A SEGUIR

Result = minimize(fun = CalculoCustoRegClas, x0 = initial_thetaClas, args = (X, Y, m, n, lbd),
                  method = 'TNC', jac = GradientRegClas);
optThetaClas = Result.x;
optJClas = Result.fun
print('Com minimize de scipy.optimize se chega a um custo optJ de {0} e optTheta {1}'.format(optJ, optTheta))

