# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:33:47 2019

@author: etienne.vanhaecke
Exercicio 7.2 PCA (Principal Component Analisys)
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
pasta="C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex7\\ex7\\"

data = loadmat(pasta+'ex7data1.mat')

XOrig = data['X']
m, n = XOrig.shape #numero de exemplos e numero de feature de cada exemplo

#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,k)))
fig1, ax1 = plt.subplots(figsize=(12,8))
scat1 = ax1.scatter(XOrig[:, 0], XOrig[:, 1], s=20, c='b', marker='o')

#Normalizacao das 2 features de X
# normalize the features
X = XOrig.copy()
meanX = np.mean(X, axis=0)
X = (X - meanX)
stdX = np.std(X, axis=0)
X = X / stdX

#Calculo da covariance matrice
XCov = 1/m*(np.dot(X.T, X))
#use the "svd" function to compute the eigenvectors  and eigenvalues of the covariance matrix. 
# perform SVD
U, S, V = np.linalg.svd(XCov)

#Draw the eigenvectors centered at mean of data. These lines show the
#directions of maximum variations in the dataset.
#Para a primeira feature
xPlt = [meanX[0], meanX[0]+1.5*S[0]*U[0,0]]
yPlt = [meanX[1], meanX[1]+1.5*S[0]*U[1,0]]
plt.plot(xPlt, yPlt)

#Para a segunda feature
xPlt = [meanX[0], meanX[0]+1.5*S[1]*U[0,1]]
yPlt = [meanX[1], meanX[1]+1.5*S[1]*U[1,1]]
plt.plot(xPlt, yPlt)

k = 1 #the top k eigenvectors para aplicar a reducao de X com o algo PCA

#Computes the reduced data representation when projecting only 
#on to the top k eigenvectors
#computes the projection of 
#the normalized inputs X into the reduced dimensional space spanned by
#the first K columns of U. It returns the projected examples in Z.
#For the i-th example X(i,:), the projection on to the k-th 
#               eigenvector is given as follows:
#                    x = X(i, :)';
#                    projection_k = x' * U(:, k);
def ProjectDate(X, U, k):
    #Inicializacao da porjectcao Z de X reduzida aos top k eigenvectors, a retornar
    Z = np.zeros((X.shape[0], k))
    
    URed = U[:, :k]
    
    Z = np.dot(X, URed)
    
    return Z

Z = ProjectDate(X, U, k)

#Recovers an approximation the 
#   original data that has been reduced to K dimensions. It returns the
#   approximate reconstruction in X_rec.
def RecoverData(Z, U, k):
    #Inicializacao da recuperacao X_rec de Z a retornar
    XRec = np.zeros((Z.shape[0], U.shape[1]))
    
    URed = U[:, :k]
    
    XRec = np.dot(Z, URed.T)
    
    return XRec
        
XRec  = RecoverData(Z, U, k)

#Se desenha a reconstrucao de X no mesmo grafico, pensando a denormalizar
#ja que o algo PCA foi aplicado sobre X normalizado
scat2 = ax1.scatter((XRec[:, 0]*stdX[0]+meanX[0]), (XRec[:, 1]*stdX[1]+meanX[1]), s=20, c='r', marker='o')


data = loadmat(pasta+'ex7faces.mat')

XOrig = data['X']
m, n = X.shape #numero de exemplos e numero de feature de cada exemplo

plt.figure('FIG2', (9, 9), clear='True')
fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, squeeze=True, num='FIG2') 
plt.xticks(()) #Para retirar o axe X
plt.yticks(()) #Para retirar o aze Y  
for i in range(10):
    for j in range(10):
        ax[i, j].imshow(XOrig[i*10+j].reshape(32, 32, order='F'), resample=True) #usar cmap='Greys' ´para variacoes de gris mas nao rende tanto bem que com o cmap default
plt.tight_layout() #Ajusta a dimensao do window a dimensao dos sub-plots
plt.show()

#Normalizacao das 2 features de X
# normalize the features
X = XOrig.copy()
meanX = np.mean(X, axis=0)
X = (X - meanX)
stdX = np.std(X, axis=0)
X = X / stdX

#Calculo da covariance matrice
XCov = 1/m*(np.dot(X.T, X))
#use the "svd" function to compute the eigenvectors  and eigenvalues of the covariance matrix. 
# perform SVD
U, S, V = np.linalg.svd(XCov)

#Se apresenta os 36 primeiros componentes principais que representam as prinicpais variacoes das fotos
plt.figure('FIG3', (9, 9), clear='True')
fig3, ax3 = plt.subplots(nrows=6, ncols=6, sharex=True, sharey=True, squeeze=True, num='FIG3') 
plt.xticks(()) #Para retirar o axe X
plt.yticks(()) #Para retirar o aze Y  
for i in range(6):
    for j in range(6):
        ax3[i, j].imshow(U.T[i*6+j].reshape(32, 32, order='F'), resample=True) #usar cmap='Greys' ´para variacoes de gris mas nao rende tanto bem que com o cmap default
plt.tight_layout() #Ajusta a dimensao do window a dimensao dos sub-plots
plt.show()

#Se projeta cada imagem sobre os 100 primeiros componentes do PCA
k = 100
Z = ProjectDate(X, U, k)

#Se remonta as fotos a parte da projecao realizada sobre os 100 primeiros componentes do PCA
XRec  = RecoverData(Z, U, k)

#Se apresenta as 100 primeiras fotos remontadas
plt.figure('FIG4', (9, 9), clear='True')
fig4, ax4 = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, squeeze=True, num='FIG4') 
plt.xticks(()) #Para retirar o axe X
plt.yticks(()) #Para retirar o aze Y  
for i in range(10):
    for j in range(10):
        ax4[i, j].imshow(XRec[i*10+j].reshape(32, 32, order='F'), resample=True) #usar cmap='Greys' ´para variacoes de gris mas nao rende tanto bem que com o cmap default
plt.tight_layout() #Ajusta a dimensao do window a dimensao dos sub-plots
plt.show()


