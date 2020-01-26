# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:42:04 2019

@author: etienne.vanhaecke
Exercicio 8.2 Coursera ML: Recomender System com Collaborative Filtering
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat


#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
pasta="C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex8\\ex8\\"

data = loadmat(pasta+'ex8_movies.mat')

#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
R = data['R']
#Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
Y = data['Y']
nM, nU = R.shape #numero de Movies (linha) e numero de Usuarios (coluna)
n = 100 #Volumetria de features X dos filmes a definir para cada film 
        #e dos pesos destas features para cada usuario

#From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): ', 
       np.mean(Y[1, R[1,:]==1]))

#We can "visualize" the ratings matrix by plotting it with imagesc
fig1, ax1 = plt.subplots(figsize=(12,8))
ax1.imshow(Y)
#ax1.imshow(Y, extent=[0, 1, 0, 1])
ax1.set_ylabel('Movies');
ax1.set_xlabel('Users');

# ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data2 = loadmat(pasta+'ex8_movieParams.mat')
X = data2['X']
theta = data2['Theta']


#  Reduce the data set size so that this runs faster
numUsers = 4
numMovies = 5
numFeatures = 3;
X = X[:numMovies, :numFeatures]
theta = theta[:numUsers, :numFeatures]
Y = Y[:numMovies, :numUsers]
R = R[:numMovies, :numUsers]



def cofiCostFunc(params, Y, R, numUsers, numMovies, numFeatures, paramLambda):
#Collaborative filtering cost function
#Returns the cost and gradient for the collaborative filtering problem.

    #Unfold the U and W matrices from params
    X = params[:numMovies*numFeatures].reshape(numMovies, numFeatures)
    theta = params[numMovies*numFeatures:].reshape(numUsers, numFeatures)
            
    # You need to return, enfilierados, the following values correctly
    J = 0;
    XGrad = np.zeros(X.shape)
    thetaGrad = np.zeros(theta.shape)

    # Instructions: Compute the cost function and gradient for collaborative filtering. 
    # Concretely, you should first implement the cost function (without regularization)
    # and make sure it is matches our costs. After that, you should implement the gradient 
    #and use the checkCostFunction routine to check that the gradient is correct. 
    #Finally, you should implement regularization.

    #1era OPCAO
    #erro = np.dot(X, theta.T)[R==1]-Y[R==1]
    #J = 1/2*np.sum(erro**2) + paramLambda/2*np.sum(X**2) + paramLambda/2*np.sum(theta**2)

    #SEGUNDA OPCAO
    erro = np.dot(X, theta.T)-Y
    J = 1/2*np.sum((erro*R)**2) + paramLambda/2*np.sum(X**2) + paramLambda/2*np.sum(theta**2)

    # Notes: X - numMovies  x num_features matrix of movie features
    #        Theta - numUsers  x num_features matrix of user features
    #       Y - numMovies x num_users matrix of user ratings of movies
    #       R - numMovies x num_users matrix, where R(i, j) = 1 if the i-th movie was rated by the j-th user

    #You should set the following variables correctly:
    #XGrad - numMovies x num_features matrix, containing the partial derivatives w.r.t. to each element of X
    #hetaGrad - num_users x num_features matrix, containing the partial derivatives w.r.t. to each element of Theta

    XGrad = np.dot((erro*R), theta) + paramLambda*X
    thetaGrad = np.dot((erro*R).T, X) + paramLambda*theta 
   
    grad = np.concatenate((XGrad.flatten(), thetaGrad.flatten()))

    return J, grad

#  Evaluate cost function
params = np.concatenate((X.flatten(), theta.flatten()))
J, grad = cofiCostFunc(params, Y, R, numUsers, numMovies, numFeatures, 0)
           
print('Cost at loaded parameters WITHOUT regularization: {0} \n(this value should be about 22.22)\n'
      .format(J))

J, grad = cofiCostFunc(params, Y, R, numUsers, numMovies, numFeatures, 1.5)
           
print('Cost at loaded parameters WITH regularization lambda 1.5: {0} \n(this value should be about 31.34)\n'
      .format(J))


# ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first add ratings that correspond to a new user that we 
# just observed. This part of the code will also allow you to put in your own ratings for the movies in our dataset!

# load movie list
movieList = open(pasta+"movie_ids.txt","r").read().split("\n")#[:-1]
# see movie list
np.set_printoptions(threshold=1000)
movieList


# Initialize my ratings
my_ratings = np.zeros((nM, 1))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4;

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2;

# We have selected a few movies we liked / did not like and the ratings we gave are as follows:
my_ratings[6] = 3
my_ratings[16]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('\n\nNew user ratings:\n')
for i in range(my_ratings.shape[0]):
    if my_ratings[i] > 0: 
        print('Rated {0} for {1}\n'.format(my_ratings[i, 0], movieList[i]))

# ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating dataset of 1682 movies and 943 users

print('\nTraining collaborative filtering...\n');

#Load data
data = loadmat(pasta+'ex8_movies.mat')
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
R = data['R']
#Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
Y = data['Y']

#Add our own ratings to the data matrix
Y = np.concatenate((my_ratings, Y), 1)
R = np.concatenate(((my_ratings!=0.), R), 1)
nM, nU = R.shape #numero de Movies (linha) e numero de Usuarios (coluna)
nF = 10 #Numero de features

#Normalize Ratings
YMean = np.sum(Y, 1) / np.count_nonzero(R, 1)
YNorm = np.zeros(Y.shape)
for i in range(nM):
    YNorm[i, R[i]==1] = Y[i, R[i]==1] - YMean[i] 

# Set initial Parameters (Theta,X)
X = np.random.randn(nM, nF)
Theta = np.random.randn(nU, nF)

paramInit = np.append(X.ravel(), Theta.ravel())

from scipy.optimize import minimize
#Uso de funcoes de optimizacao da biblioteca scipy.optimize
#Metodo CG
#Method CG uses a nonlinear conjugate gradient algorithm by Polak and Ribiere, a variant of 
#the Fletcher-Reeves method described in 5 pp. 120-122. Only the first derivatives are used.
paramLambda = 10
#Com jac a True, o otimizador espera que a funcao objetivo, apontada por fun, retorna o gradient ademais do custo 
Result = minimize(method = 'CG', fun = cofiCostFunc, x0 = paramInit, 
                      args = (Y, R, nU, nM, nF, paramLambda), jac = True) 
paramOtim = Result.x
JOtim = Result.fun

#Se desenrola a lista de parametro em X e theta
X = paramOtim[:nM*nF].reshape(nM, nF)
theta = paramOtim[nM*nF:].reshape(nU, nF)

print('Com metodo de otim CG de scipy.optimize se chega a um custo JOpt de {0}'.format(JOtim))

#Predict rating para mim (1era coluna), lembrando de readicionar a media de notacao de cada filme
p = np.dot(X, theta[0, :].T) + YMean

#Se recupera o index dos 10 filmes com maior nota prevista 
idxTop = p.argsort()[::-1][0:10]
print("Top recommendations for you:\n")
for i in range(10):
    print("Predicting rating {0} para o filme de ID {1} titulo {2}"
          .format(p[idxTop[i]], idxTop[i], movieList[idxTop[i]]))