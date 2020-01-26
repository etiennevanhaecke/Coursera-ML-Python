# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:31:38 2019

@author: etienne.vanhaecke
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:52:37 2019

@author: etienne.vanhaecke
Exercicio 4 do curso de ML de Coursera: Treinamento dos parametros (pesos) de Rede Neuronal por back-propagation, para deducao digito escrito manualmente

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
data = loadmat("C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex3\\ex3\\ex3data1.mat", variable_names=('X', 'y'))

#Parametros theta ja treinados para verificar a funcao custo
thetaEx = loadmat("C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex3\\ex3\\ex3weights.mat", variable_names=('Theta1', 'Theta2'))
thetaEx1 = thetaEx['Theta1'] #Parametros (=pesos) da funcao mapping para passar da camada de entrada formada de 401 features (400 da foto+constante) ate a camada escondida formada de 25 unidades de ativacao
thetaEx2 = thetaEx['Theta2'] #Parametros (=pesos) da funcao mapping para passar da camada escondida formada de 26 features (25 unidades de ativacao+constante) ate a camada output formada das 10 classes de digito (0 a 9)


X=data['X']
m, n = X.shape #Numeros de exemplos de fotos de digitos e de feature (intensidade de gris de cada pixel da imagem do digito)
#Se adiciona o bias (unidade 0 da camada de entrada, x0)
X=np.insert(arr=X, obj=0, values=1, axis=1) #Outra solucao para adicionar a coluna de 1 em 1era posicao (se pode usar tambem de hstack e concatenate de numpy)

YOrig=data['y'] #Recuperacao do label (digito de 0 a 9), com 0 representado por 10
YLabel=YOrig.copy()
YLabel[YLabel[:,0]==10,0]=0 #Se transforma 0 10 em 0, para poder realizar o One Hot Encoder considerando a posicao do index 
                          #onde se encontra o 1 como o label do digito (index 0 = label 0, index 2 = label 2)
clas = np.unique(YLabel) #Vector horizontal (1 dimensao) dos diferentes labels 
k=len(clas) #Numero de classes (unidades da camada output)

#Precisa transformar o label em serie de 0 e 1. a dimensao desta serie sendo o numero de label e
#uma unica posicao da serie tera o valor 1 para representar o digito correspondente.
#Como se transformou o label 10 em label 0, o index do 1 ira representar o label 
#(1 no index 0 = label 0, 1 no index 9 = label 9)
#Existem funcoes prontas para realizar esta transformacao como
#LabelBinarize e OneHotEncoder de sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
#Exemplo de codagem com LabelBinarizer (recomendado na doc de OneHotEncoder)
lb = LabelBinarizer()
Y = lb.fit_transform(YLabel)
#Exemplo de codagem com OneHotEncoder (recomendado na doc de OneHotEncoder)
oh = OneHotEncoder(drop = None, sparse=False, categories='auto') #A opcao drop retira um valor para evitar redondancia (o label retirado esta representado com uma codificacao toda a 0)
Y = oh.fit_transform(YLabel)
#Loop para realizar esta codagem de scratch
print('Codificaacao dos {0} labels do jogo de dado em vectores de {1} dimensao com a posicao do 1 representando o valor do label'.format(m, k))
Y=np.zeros((m, k)) #Inicializacao a 0 da nova variavel dos labels sobre as novas dimensoes da codificacao (m linhas de n colunas)
for i in range(k):
    print('Codificacao do label {0} com o unico 1 no index de coluna {1}'.format(clas[i], i))
    Y[YLabel[:, 0]==clas[i], i] = 1

e1 = 25 #Numero de unidade de ativacao da unica camada escondida da rede neuronal (em posicao l = 2 entre a camada input e a camada output)
lbd = 1 #Fator de regularizacao para evitar o overfitting penalisando as hipoteses com maior soma de parametros theta
epsInitTheta = 0.12 #Epsilon usado para inicializar os pesos arredor de -/+ o valor deste epsilon
#Epsilon usado para checkar o gradient de custo de um vector de peso calculado por back propagation calculando um slot da linha de custo usando o vector de peso ajustado iterativamente a cada index de -/+ o valor deste epsilon
epsGradCheck = 1e-4 #Outra escritura np.power(10., -4)
fracaoJogo = 0.01 #Fracao de uso do jogo de dado (as fotos dos digitos) para o check do gradient calculado via backpropagation via uma aproximacao usando os custos a +/- epsilon do theta do gradient a checkar

#Funcao para inicializar os pesos de todas as unidades da rede neuronal de maneira aleatoria, com valores distribuidos aredor de 0
#Esta inicializacao aleatoria explica porque o resultado da aprendizagem pode ficar diferente, caso a curba de evo do custo em funcao
#dos pesos nao esta convexo (presenca de varios pontos baixos), permitindo a selecao de ponto baixo diferente em funcao do ponto inicial
#de pesquisa (determinado pelo valor inicial dos pesos de propagacao do sinal entre as unidades)
def RandInicializeWeight(eps, n, e1, k):
    sizeTheta1 = (n+1) * e1 #Tamanho do vectpr de peso para passar da camada input ate a camada intermediaria
                            #Igual ao numero de feature de X+1 (para considerar o bia) mulitplicado pelo numero de unidade da camada escondida
    sizeTheta2 = (e1+1) * k #Tamanho do vectpr de peso para passar da camada escondida ate a camada output
                            #Igual ao numero de unidade da camada escondida +1 (para considerar o bia) mulitplicado pelo numero de classe da camada output
                                               
    theta=np.random.random(size=(sizeTheta1+sizeTheta2)) *2*eps - eps
    print('theta: num pesos {0} de min {1} e max {2}'.format(len(theta), np.min(theta), np.max(theta)))

    return theta    
  
#Funcao de ativacao (g()) aplicada sobre a soma dos valores de entrada (z) correspondendo aos valores output liberado pelas unidades
#da camanda anterior multiplicadas pelos pesos entre estas unidades da camada anterior e a unidade destino da camada seguinte.
def Sigmoid(z):
    z = np.array(z) #Caso z seja passado como lista
    return 1 / (1 + np.exp(-z))

#Derivado (=gradient) da funcao de ativacao encima, usado para calcular a fracao do erro total (calculado com a funcao custo) decorrente de cada unidade
def SigmoidGradient(z):
    z = np.array(z) #Caso z seja passado como lista
    return Sigmoid(z) * (1-Sigmoid(z))


#Previsao do digito de cada foto do jogo de dados passado (pode estar um vector com as features de um unico digito
#como pode estar uma matrice com as features de varias fotos, usando os parametros da rede neuronal ja treinada.
#aplicando a forward propagation (desde a 1era camada de input ate a ultima camada de output)
def Predict(X, theta1, theta2, m):
    #Se X esta um vector linha (um unico exemplo de digito com as suas features)
    if X.ndim == 1:
        X = X.reshape(1, len(X)) #Para esta compativel com os calculos seguintes, se passa ele em array de uma linha
    #Calculo do valor de entrada a2 de cada uma das 25 unidades de ativacao (UA) da camada escondida
    #por uma multiplicacao matricial de cada feature da camada de entrada pelos pesos de cada UA (theta1)
    z2 = np.dot(X, theta1.T)
    a2 = Sigmoid(z2) #Se usa o transverse de theta1 para ter os pesos de cada UI (401) como coluna
                             #Cada coluna de a2 vai corresponder ao valor das 25 UA por exemplo de digito
    
    #Se completa os valores das features das UA com a coluna de 1 representando o bias (1era coluna)
    a2=np.hstack((np.ones((m, 1)), a2)) #CUIDADO coluna de 1 em primeira posicao e nao ultima
    #Calculo do valor de entrada a2 de cada uma das 10 classes (digito de 0 a 9) da camada output
    #por uma multiplicacao matricial de cada feature da camada esconida pelos pesos de cada classe (theta2)
    z3 = np.dot(a2, theta2.T)
    H = Sigmoid(z3) #Se usa o transverse de theta2 para ter os pesos de cada UI (26) como coluna
                              #Cada coluna de a2 vai corresponder ao valor das 10 classes por exemplo de digito
    
    return np.argmax(H, axis=1), H, z3, a2, z2 #Se retorna para cada exemplo digito do jogo de dados a probabilidade de cada classe digito
                                       # e a classe digito de maior probabilidade.

#Cost Funcao, em funcao pesos/parametros de cada unidade da rede neuronal
def NnCostFunction(theta, X, Y, m, n, e1, k, lbd, modo=0, flagReg=1, flagGrad=1):
    #Como os parametros(=pesos) estao passados enfileirados, se redimensional eles para cada funcao mapping de camada l-1 para camada l
    #Como a nossa rede esta formada de 3 camadas, se "desenfileira" em theta1 dos pesos entre a camada 1 inbound e a unica camada escondida
    #de posicao 2 (shape linha e1 de 25 e coluna n+1 de 401) e theta 2 dos pesos entre a camada escondida e a camada outout (ahape k linhas e e1+1 colunas)
    theta1 = theta[0:(n+1)*e1].reshape(e1, n+1) #(25, 401) Recuperacao dos pesos de transicao da camada in ate a camada escondida
    theta2 = theta[(n+1)*e1:].reshape(k, e1+1) #(10, 26) Recuperacao dos pesos de transicao da camada escondida ate a camada out
    
    Clas, H, z3, a2, z2 = Predict(X, theta1, theta2, m) #Calculo da probabilidade das diferentes classes da camada output
    
    #So para verificar o custo com os theta recuperados do site, que foram calculados com MatLab de indice iniciando por 1
    #e entretanto com a unidade output da classe 0 colocada em decima posicao com o label 10, se passa a ultima coluna em 1era posicao
    #empurrando de uma posicao as outras colunas de classe de 1 a 9 (assim o index da coluna corresponde ao label do digito para comparar contra Y)
    if modo == 1:
        H=np.roll(H, 1, axis=1) #NAO CHAMAR (usar modo = 0) durante o treinamento do modelo
    
    #Calculo com 2 formulas do custo de cada um dos 10 classificadores fazendo uma soma do desvio de cada exemplo digito (os 5000 de X) 
    #Por soma numpy da array completa (sobre as 2 dimensoes de m exemplo de X e k classe de Y)
    J1 = 1/m * np.sum(( - Y*np.log(H)) - (1-Y)*(np.log(1-H))) 
    #Pela suma da probabilidade das k classes resultando da multiplicacao matricial (TALVE MAIS RAPIDO) recuperando so a diagonal que corresponde ao dot da linha e de coluna de mesmo index
    J2 = np.sum(np.diagonal(1/m * ( - np.dot(Y.T, np.log(H)) - np.dot((1-Y).T, np.log(1-H))))) 
                                                                                                            
    #np.einsum('ij,ji->i', a, b) #Solucao possivel para o calculo so da diagonal (dot de linha com coluna de mesmo index) sem realizar todo o calculo matricial dos 2 arrays
    #Regularizacao com lambda usando o metodo Ridge
    if (flagReg == 1): #Se regularizacao
        J = J2 + (lbd/(2*m) * ( np.sum(theta1[:, 1:]**2)+np.sum(theta2[:, 1:]**2) )) #CUIDADO a indicar em que axe realizar a soma, senao np.sum soma todos os elementos sobre os diferentes axes  
    else: #Se nao tem regularizacao
        J = J2
    
    #Inicio da back propagation do custo da camanda de saida
    #Feito exemplo a exemplo do jogo de dados em um loop
    DELTA1 = np.zeros(theta1.shape)  # (25, 401) 
    DELTA2 = np.zeros(theta2.shape)  # (10, 26)
    if flagGrad == 1:
        for i in range(m): #Para cada exemplo do jogo de dados
            d3 = H[i]-Y[i] #delta da camada outbound
            d3 = d3.reshape(d3.size, 1) #Para ter a shape esperada de (10,1)
        
            #Para calcular o delta da camada escondida 2, com o valor de ativacao
            #z2 desta camada, se adiciona 1 na frente de z2 para considerar o bias que esta presente
            #em theta2 (DUVIDA SE TEM OU NAO CONSIDERAR O BIAS NA BACK-PROPAGATION)
            z2i = z2[i].reshape(1, z2[i].size)  #Shape esperado (1, 25)
            z2i = np.insert(arr=z2i, obj=0, values=1, axis=1) #Shape esperado (1, 26)
            d2 = np.dot(theta2.T, d3) * SigmoidGradient(z2i.T)
            d2 = d2.reshape(d2.size, 1) #Para ter a shape esperada de (26,1)

            DELTA1 = DELTA1 + np.dot(d2[1:], X[i].reshape(1, X[i].size)) #Shape esperada (25, 401)        
            DELTA2 = DELTA2 + np.dot(d3, a2[i].reshape(1, a2[i].size)) #Shape esperada (10, 26)        

    grad1 = DELTA1/m
    grad2 = DELTA2/m     
    if (flagReg == 1): #Se regularizacao
        grad1[:, 1:] = grad1[:, 1:] + lbd/m * theta1[:, 1:]
        grad2[:, 1:] = grad2[:, 1:] + lbd/m * theta2[:, 1:]       
   
    
    #Gradient enfilieiradp
    grad = np.concatenate( (grad1.ravel(), grad2.ravel()) )
        
    return J, grad #Retorno de um vector com o custo de cada classificador (um por digito de 0 a 9)
                   #E o gradient de cada peso de theta1 e theta2

#Se verkifica o custo da rede neuronal com os parametros theta recuperados da rede ja treinada
#Como a nossa funcao custo espera um unico parametro de peso, se enfiliera os pesos das 2 transicoes entre camada
thetaExEnfil=np.hstack((thetaEx1.ravel(), thetaEx2.ravel()))
thetaExEnfil.shape #(10285,)

JExSemRegul, NaoUsado = NnCostFunction(thetaExEnfil, X, Y, m, n, e1, k, lbd, 1, 0, 0) #0.2876 esperado SEM regularizacao e usando o modo 1 indicando que os pesos da classe 0 estao em ultima posicao e SEM calc Grade
print('Custo JExSemRegul com os parametros thetaEx esta {0}'.format(JExSemRegul))

JExComRegul, NaoUsado = NnCostFunction(thetaExEnfil, X, Y, m, n, e1, k, lbd, 1, 1, 0) #0.383 esperado COM regularizacao e usando o modo 1 indicando que os pesos da classe 0 estao em ultima posicao e SEM calc Grade
print('Custo JExSemRegul com os parametros thetaEx esta {0}'.format(JExComRegul))

#Inicializacao aleatoria dos parametros theta das funcoes de transicao entre as 3 camadas, fluturando entre
#menos epsilon e mais epsilon, no formato de um vector horizontal
theta = RandInicializeWeight(epsInitTheta, n, e1, k)
thetaInit = theta.copy()
#Calculo do custo e do gradient com estes pesos iniciais
J, grad = NnCostFunction(theta, X, Y, m, n, e1, k, lbd, flagReg=1) #SEM deslocamento do label de 0. COM regularizacao e com calculo da grade
print('Custo J com os parametros theta iniciais aleatorios esta {0}'.format(J))


#Calculo de uma aproximacao do derivado (=gradient) do custo (mais simple mas bem mais lenta que a back-propagacao), definindo o slote
#de cada peso calculando o custo aredor deste peso (de +/- epsilon)
def ComputeNumericalGradient(theta, X, Y, m, n, e1, k, eps):
    print('ComputeNumericalGradient para check gradient back propagation com epsilon {0}'.format(eps))
    t = len(theta)
    arrayEps = np.zeros(t) #Vector so de zero do tamanho do vector de peso 
    gradCheck = np.zeros(t) #Vector so de zero do tamanho do vector de peso 
    
    for i in range(t): #Para cada peso  
        arrayEps[i] = eps      #Na posicao i deste array se adiciona epsilon
        #Adicao de epsilon no peso de index i, deixando os outros pesos inalterados
        thetaEpsPlus = theta + arrayEps
        #Calculo do custo com este vector de peso ajustado no index i com mais epsilon
        JEpsPlus, NaoUsado = NnCostFunction(thetaEpsPlus, X, Y, m, n, e1, k, lbd, modo=0, flagReg=0, flagGrad=0) #SEM deslocamento do label de 0. COM regularizacao e SEM calculo da grade
        #Retirada de epsilon no peso de index i, deixando os outros pesos inalterados
        thetaEpsMenos = theta - arrayEps
        #Calculo do custo com este vector de peso ajustado no index i com mais epsilon
        JEpsMenos, NaoUsado = NnCostFunction(thetaEpsMenos, X, Y, m, n, e1, k, lbd, modo=0, flagReg=0, flagGrad=0) #SEM deslocamento do label de 0. COM regularizacao e SEM calculo da grade

        #Calculo do gradient do peso de posicao i a parte destes custos de theta ajustado de epsilon
        gradCheck[i] = (JEpsPlus-JEpsMenos) / (2*eps)

        arrayEps[i] = 0 #Se reinicializa a 0 a posicao i deste array antes da proxima iteracao do elemento de theta
        
    return (gradCheck) #Retorno deste gradient aproximado para check do gradient calculado pela back propagation

from random import seed, sample
seed() #Inicializa o generator de numero aleatorio a parte da hora do sistema
    
#Faz o check que os gradientes calculados pela funcao encima estao coerentes comparando eles com uma aproximacao do calculo do derivado
#obtida tracando uma linha entre os pontos da curba de evo custo em funcao peso de X peso verificado - epsilon (ordem 0,004) e + epsilon.
def CheckNNGradient(grad, theta, X, Y, m, n, e1, k, eps, fracaoJogo):
    #Visto a lentidao do metodo aproximativado de calculo do gradient que necessita 2 chamadas 
    #a funcao de custo, se vai limitar o jogo de dados a uma porcao do jogo total
    idxRdm = sample(range(m), int(fracaoJogo*m)) #Selecao aleatoria de uma fracao index dentro do range do numero de exemplos (5000) do jogo de dados
    XGradCheck = X[np.array(idxRdm)]    #Se monta o array das features de check considerando a mostragem aleatoria de exemplos
    YGradCheck = Y[np.array(idxRdm)]    #Se monta o array das features de output considerando a mostragem aleatoria de exemplos
    mGradCheck=len(XGradCheck)          #Tamanho do jogo de dados reduzido
    
    #Calculo do gradient aproximativo para check
    gradCheck = ComputeNumericalGradient(theta, XGradCheck, YGradCheck, mGradCheck, n, e1, k, eps)

    #Comparacao visual do gradient calculado via back-propagation (vai servir ao treinamento visto a perf dele) contra
    #este calculado via slote de custo de theta entre +/- epsilon nao apropriado ao treinaento devido a perf muito baixa:
    compGrad = np.hstack((grad.reshape(grad.size, 1), gradCheck.reshape(gradCheck.size, 1)))
    
    return (compGrad)

compGrad = CheckNNGradient(grad, theta, X, Y, m, n, e1, k, epsGradCheck, fracaoJogo)
'''
DIFERENCA TALVEZ LIGADA AO PESO DO BIAS CONSIDERADO NA BACKPROPAGATION.
A REVER OLHANDO O FONTE DE COURSERA
TENTAR CALCULAR OS GRADIENT POR DOT PRODUCT DOS EXEMPLOS EM VEZ DE REALIZAR UM LOOP
'''     

#Uso de funcoes de optimizacao da biblioteca scipy.optimize
from scipy.optimize import minimize
Result = minimize(fun = NnCostFunction, x0 = thetaInit, args = (X, Y, m, n, e1, k, lbd),
                  method = 'TNC', jac = True, options={'maxiter': 2500, 'disp':True});
optTheta = Result.x;
optJ = Result.fun
print('Com minimize de scipy.optimize se chega a um custo optJ de {0}'.format(optJ))

optTheta1 = optTheta[0:(n+1)*e1].reshape(e1, n+1) #(25, 401) Recuperacao dos pesos de transicao da camada in ate a camada escondida
optTheta2 = optTheta[(n+1)*e1:].reshape(k, e1+1) #(10, 26) Recuperacao dos pesos de transicao da camada escondida ate a camada out
                          
#Verificacao da acuracidade de cada classe da rede neuronal sobre o jogo de dados inteiro
print('Acuracidade a rede neuronal sonre o jogo de dados de {0} exemplos '.format(m))

classe, H, z3, a2, z2  = Predict(X, optTheta1, optTheta2, m) #Digito previsto para cada linha exemplo de X

compClassecomYLabel = np.hstack((classe.reshape(m, 1), YLabel)) #Comparacao contra a classificacao real colocando o previsto e o label em um mesmo array
for i in range(10):
    #print(i)
    tot = len(YLabel[YLabel==i])
    #print(tot)
    ok = np.sum(classe.reshape(m, 1)[YLabel==i]==YLabel[YLabel==i])
    #print(ok)

    dig = i
        
    print('Digito {0}: {1} classificados corretamente sobre os {2} do jogo - {3}%'
          .format(dig, ok, tot, np.round(ok/tot*100, 2)))

#Visualizacao dos pesos da camada escondida, por grid igual a das features de entrada
#dos digitos (20, 20) para cada uma das unidades de ativacao da camada escondida
plt.close('all') #Fechamento de todas as figuras abertas
fig1 = plt.figure(figsize=(8, 6))
plt.xticks(()) #Para retirar o axe X
plt.yticks(()) #Para retirar o aze Y
plt.title('Visualizacao interna com optTheta1 da camada escondida por unidade de ativacao',
          fontdict={'fontsize': plt.rcParams['axes.titlesize'], 'fontweight' : plt.rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': 'center'})
for i in range(25):
    ax = fig1.add_subplot(5, 5, (i+1))  
    plt.xticks(()) #Para retirar o axe X
    plt.yticks(()) #Para retirar o aze Y    
    ax.imshow(optTheta1[i, 1:].reshape(20, 20, order = 'F'), cmap='Greys')
    ax.set_title('UA {0}'.format(i))
plt.tight_layout() #Ajusta a dimensao do window a dimensao dos sub-plots
plt.show()

fig2 = plt.figure(figsize=(8, 6))
plt.xticks(()) #Para retirar o axe X
plt.yticks(()) #Para retirar o aze Y
plt.title('Visualizacao interna com thetaEx1 da camada escondida por unidade de ativacao',
          fontdict={'fontsize': plt.rcParams['axes.titlesize'], 'fontweight' : plt.rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': 'center'})
for i in range(25):
    ax = fig2.add_subplot(5, 5, (i+1))  
    plt.xticks(()) #Para retirar o axe X
    plt.yticks(()) #Para retirar o aze Y    
    ax.imshow(thetaEx1[i, 1:].reshape(20, 20, order = 'F'), cmap='Greys')
    ax.set_title('UA {0}'.format(i))
plt.tight_layout() #Ajusta a dimensao do window a dimensao dos sub-plots
plt.show()