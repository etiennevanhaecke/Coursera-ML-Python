# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:42:04 2019

@author: etienne.vanhaecke
Exercicio 8.1 Coursera ML: Anomaly Identification
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn

#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
pasta="C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex8\\ex8\\"

data = loadmat(pasta+'ex8data1.mat')

X = data['X']
m, n = X.shape #numero de exemplos e numero de feature de cada exemplo
XVal = data['Xval']
YVal = data['yval']


#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,k)))
fig1, ax1 = plt.subplots(figsize=(12,8))
scat1 = ax1.scatter(X[:, 0], X[:, 1], s=20, c='b', marker='x')
ax1.set_xlabel('Latency (ms)');
ax1.set_ylabel('Throughput (mb/s)');

#Calculo das propriedades da distribuicao Gossian do X: Media e variancia
def EstimateGaussian(X, m):
    mu = 1/m*np.sum(X, 0)               #Media da serie
    sigma2 = 1/m*np.sum((X-mu)**2, 0)   #Variancia da serie
    
    return mu, sigma2
    
mu, sigma2 = EstimateGaussian(X, m)

#Calculo da probability density function of the examples X under 
#the multivariate gaussian distribution with parameters mu and sigma2 
p = mvn.pdf(X, mu, sigma2)

def VisualizeFit(X, mu, sigma2, ax1):
#Visualize the dataset and its estimated distribution.
#This visualization shows you the probability density function of the
#Gaussian distribution. Each example has a location (x1, x2) that 
#depends on its feature values.

    #SE monta um grid pelo qual tera calculo do PDF para cada ponto
    #O retorno X1 corresponde ao coordenado X de cada ponto e X2 do coordenado Y
    X1, X2 = np.meshgrid(np.arange(0, 35, 0.5), np.arange(0, 35, 0.5), sparse=False) 
    #Se monta uma serie de array das coordenadas X, Y de cada ponto do grid
    #Para isso, tem que deixar as coordenadas como vectores com ravel,
    #juntar eles em tuplet de coordenados com zip, transformar em list
    #e por fim transformat em um array 2D (1era D linha cada pixel e segunda D colunas com coordenadas)
    X12 = np.array(list(zip(X1.ravel(), X2.ravel())))
    #Calculo do PDF de cad ponto do grid, usando os atributos media e variancia da serie exemplo
    Z = mvn.pdf(X12, mu, sigma2)
    #Se redimensiona o PDF para o shape do grid
    Z = Z.reshape(X1.shape)
    #Se desenha sobre o grafico dos exemplos as linhas de contour em funcao do PDF
    ax1.contour(X1, X2, Z, np.power(10., np.arange(-20, 0, 3)))
    
VisualizeFit(X, mu, sigma2, ax1)

def SelectThreshold(YVal, pVal): 
#finds the best threshold to use for selecting outliers based on the 
#results from a validation set (pval) and the ground truth (yval).

    bestEpsilon = 0;
    bestF1 = 0;
    F1 = 0;

    stepsize = (max(pVal) - min(pVal)) / 1000
    for epsilon in np.arange(min(pVal), max(pVal), stepsize):
        
        #Compute the F1 score of choosing epsilon as the threshold and place
        #the value in F1. The code at the end of the loop will compare the 
        #F1 score for this choice of epsilon and set it to be the best epsilon
        #if it is better than the current choice of epsilon.

        #Se conta o TP (True Positive), FP (False Positive) e o FN (False Negative) 
        #de Xval em funcao do pVal e do epsilon
        #True Positive os exemplos de label True de probability density function 
        #(PDF) inferior ao limite testado
        TP = np.sum(pVal[YVal[:, 0]==1] < epsilon) 
        #False Negative os exemplos de label True de PDF >= ao limite testado
        FN = np.sum(pVal[YVal[:, 0]==1] >= epsilon)
        #False Positive os exemplos de label False de PDF < ao limite testado
        FP = np.sum(pVal[YVal[:, 0]==0] < epsilon)
        
        #Calculo da PRECision e do RECall
        PREC = TP / (TP+FP)
        REC = TP / (TP+FN)
        
        #Calculo do indicador F1
        F1 = (2*PREC*REC) / (PREC+REC)
        
        #Conservacao deste limite se o score F1 esta o melhor
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1

pVal = mvn.pdf(XVal, mu, sigma2)
bestEpsilon, bestF1 = SelectThreshold(YVal, pVal)

print('Best epsilon found using cross-validation: ', bestEpsilon)
print('Best F1 on Cross Validation Set: ', bestF1);
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)')

scat2 = ax1.scatter(X[p<bestEpsilon, 0], X[p<bestEpsilon, 1], 
                    s=80, c='r', marker='o', alpha=0.6)
scat2.set_label('Anomalias')
fig1.show()

#================== Part 4: Multidimensional Outliers ===================
#We will now use the code from the previous part and apply it to a 
#harder problem in which more features describe each datapoint and only 
#some features indicate whether a point is an outlier.
#

#Loads the second dataset. You should now have the
#variables X, Xval, yval in your environment
data = loadmat(pasta+'ex8data2.mat')

X = data['X']
m, n = X.shape #numero de exemplos e numero de feature de cada exemplo
XVal = data['Xval']
YVal = data['yval']

#Apply the same steps to the larger dataset
mu, sigma2 = EstimateGaussian(X, m);

#Training set 
p = mvn.pdf(X, mu, sigma2)

#Cross-validation set
pVal = mvn.pdf(XVal, mu, sigma2)

#Find the best threshold
bestEpsilon, bestF1 = SelectThreshold(YVal, pVal)

print('Best epsilon found using cross-validation: ', bestEpsilon)
print('Best F1 on Cross Validation Set: ', bestF1)
print('   (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of 0.615385)')
print('# Outliers found: ', sum(p < bestEpsilon))
