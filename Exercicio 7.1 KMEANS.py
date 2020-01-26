# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:33:47 2019

@author: etienne.vanhaecke
Exercicio 7.1 KMeains 
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
pasta="C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex7\\ex7\\"

data = loadmat(pasta+'ex7data2.mat')

X = data['X']
m, n = X.shape #numero de exemplos e numero de feature de cada exemplo

# Select an initial set of centroids
k = 3 # 3 Centroids
centroidsInit = np.array([[3,3], [6,2], [8,5]])

#Selecao aleatoria dos 3 clusters de inicio do algo de KMean
centroidsInit = X[np.random.randint(0, X.shape[0], 3)]

#computes the centroid memberships for every example
#returns the closest centroids in idx for a dataset X where each row is a single example.
# idx = m x 1 vector of centroid assignments (i.e. each entry in range [1..K])
def FindClosestCentroids(X, centroids, m, k):
    #Inicializacao do vector de retorno do indice do centroid
    #mais perto de cada exemplo
    idx = np.empty(m)
    
    #PAra cada exemplo do jogo de dados
    for i in range(m):
        dist = np.empty(k) #Inicializacao das distancias do exemplo i tratado ate cada centroide
        #Para cada centroid
        for j in range(k):
            #Calculo da distancia deste centroid j ate o exemplo tratado i
            dist[j] = np.sum((X[i, :] - centroids[j, :])**2)
            #print('Distancia de X index {0} ate centroid {1} esta {2}'
            #      .format(i, j, dist[j]))
        #Outra forma de calcular a distancia por calculo matricial
        dist = np.sum((X[i] - centroids) ** 2, 1)
        
        #Informacao do index do centroid mais perto do exemplo
        idx[i] = np.argmin(dist)
            
    return idx

#Calculo inicial do centroid de cada exemplo a parte dos centroids definidos ao inicio do programa
idx = FindClosestCentroids(X, centroidsInit, m, k)
print(idx[0:3])
print('\n(the closest centroids should be 0, 2, 1 respectively)\n');

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,k)))
fig1, ax1 = plt.subplots(figsize=(12,8))
def PlotProgresskMeans(X, centroids, centroidsAnt, idx, ax, k, i):
    scat1 = ax.scatter(X[:, 0], X[:, 1], s=20, c=idx, marker='o')
    #Se nao se trata da primeira iteracao de agrupamento por cluster
    if (centroidsAnt.size > 0):
        #Para cada centroid
        plt.plot([centroidsAnt[:, 0], centroids[:, 0]],
                 [centroidsAnt[:, 1], centroids[:, 1]])
        #Redesenha o centroid anterior com um tamanho menor que o novo centroid
        ax.scatter(centroidsAnt[:, 0], centroidsAnt[:, 1], c=np.array([0, 1, 2]), s=20, marker='x')
    #Desenho do novo centroid de cada cluster    
    scat2 = ax.scatter(centroids[:, 0], centroids[:, 1], c=np.array([0, 1, 2]), s=100, marker='x')

        
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scat2.legend_elements(),
                    loc="lower left", title="Clusters")
    ax.add_artist(legend1)
    ax.set_title("Cluster depois da iteracao {0} do KMean".format(i+1))
    
    #Se espera que o ususario aperta um teclado e se deixa o tempo de
    #de redesenhar o grafico
    plt.pause(0.05) 
    plt.draw()
    #input("Press the <ENTER> key to continue...")
    
    #Se suprime o ponto do novo centroid para fazer aparecer ele na proxima iteracao de KMean
    #com um tamanho menor para identificar facilmente o novo centroid da proxima iteracao
    scat2.remove()    

PlotProgresskMeans(X, centroidsInit, np.array([]), idx, ax1, k, 0)

#returns the new centroids by computing the means of the  %data points assigned to each centroid.
#It is %   given a dataset X where each row is a single data point, a vector idx of centroid assignments
# (i.e. each entry in range [1..K]) for each example, and k, the number of centroids.
def computeCentroids(X, idx, k, n):
    centroids = np.empty((k, n)) #Inicializacao dos novos valores dos centroids
    #Para cada centroid
    for i in range(k):
        #Se calcula os novos valores em funcao do jogo de dados associado a este centroid
        centroids[i] = np.mean(X[idx==i], 0)

    return centroids

centroidsAnt = centroidsInit
#Se calcula o novo centroid de cada cluster
centroids = computeCentroids(X, idx, k, n)

print(centroids)

print('\n(the centroids should be\n');
print('   [ 2.428301 3.157924 ]\n');
print('   [ 5.813503 2.633656 ]\n');
print('   [ 7.119387 3.616684 ]\n\n');

#Se desenha a evolucao dos clusters com os novos centroids
PlotProgresskMeans(X, centroids, centroidsAnt, idx, ax1, k, 0)
plt.pause(0.05) 
plt.draw()

#input("Press the <ENTER> key to continue...")

#Execucao do algo KkMeans o numero de vezes indicado no param maxIters,
#iniciando a parte do centroid passado em parametro
def RunkMeans(X, centroidsStart, maxIters):
    centroids = centroidsStart
    #Para o numero de iterencia passado em parametro
    for i in range(maxIters):
        #Se determina o cluster de cada exemplo do jogo de dados
        idx = FindClosestCentroids(X, centroids, m, k)
        #Se conserva a posicao dos centroids anteriores
        centroidsAnt = centroids
        #Se calcula o novo centroid de cada cluster
        centroids = computeCentroids(X, idx, k, n)
        #Se desenha a evolucao dos clusters com os novos centroids
        PlotProgresskMeans(X, centroids, centroidsAnt, idx, ax1, k, i)
                
    return centroids, idx
        
# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
maxIters = 10;
centroids, idx = RunkMeans(X, centroids, maxIters)
print('\nK-Means Done.\n\n');

plt.show()

#Apliacao das funcoes K-Means sobre uma foto
#Diferentes solucoes para carga a foto

# load and show an image with Pillow
from PIL import Image
# load the image
image = Image.open(pasta+'bird_small.png')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image.show()


# load image and convert to and from NumPy array
from PIL import Image
from numpy import asarray
# load the image
image = Image.open(pasta+'bird_small.png')
# convert image to numpy array
data = asarray(image)
# summarize shape
print(data.shape)
# create Pillow image
image2 = Image.fromarray(data)
# summarize image details
print(image2.format)
print(image2.mode)
print(image2.size)

# load image and convert to and from NumPy array
from PIL import Image
from numpy import asarray
# load the image
image = Image.open(pasta+'bird_small.png')
# convert image to numpy array
data = asarray(image)
# summarize shape
print(data.shape)

# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
data = image.imread(pasta+'bird_small.png')
# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
pyplot.figure('FIG1', (15, 8), clear='True')
fig, ax = pyplot.subplots(nrows=1, ncols=2, sharex=True, sharey=True, squeeze=True, num='FIG1') 
ax[0].imshow(data)
pyplot.show()

#Se agrupa todos os pixels da foto em 16 cluster em funcao da cor RGB
A = data
#Se redimensiona a imagrm vom ums dimensao 2
A = A.reshape(A.shape[0]*A.shape[1], A.shape[2])
k = 16 #os 16 clusters de sgrupamento de todos os pixels da foto
m, n = A.shape #m o numero de pixel e n o numero de intensidade de cor RGB por pixel

#Selecao aleatoria de 16 pixels como centroid de cada um dos 16 clusters de inicio do algo de KMean
centroids = A[np.random.randint(0, A.shape[0], k)]

#Se aplica 10 iteracoes do algo KMean
for i in range(10):
    #Se determina o cluster de cada exemplo do jogo de dados
    idx = FindClosestCentroids(A, centroids, m, k)
    #Se calcula o novo centroid de cada cluster
    centroids = computeCentroids(A, idx, k, n)

#Se remonts s foto atribuindo a cada pixel os valores RGB de um dos 16 centroids ao qual ele foi associado
#E repassando a foto em 3 dimensoes
B = centroids[idx.astype(int)].reshape(128, 128, 3)
ax[1].imshow(B)
pyplot.show()

