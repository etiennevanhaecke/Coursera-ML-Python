# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:54:39 2019

@author: etienne.vanhaecke
SMV para classificacao SPAM

"""

#%matplotlib #A RODAR UMA VEZ E DEPOIS A COMENTAR PARA OS GRAFICOS APARECER AFORA DA CONSOLE

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.io import loadmat
import re
import nltk
    
#Importacao dos dados do arquivo MAT dentro de uma array numpy. CUIDADO a usar o escape \ antes do \ entre as pastas senao \ interpretado como escape
pasta="C:\\Users\\etienne.vanhaecke\\OneDrive\\Documentos\\IA\\Coursera ML\\machine-learning-ex6\\ex6\\"

file = open(pasta+"emailSample1.txt", "r")
sample = file.read()
#print(sample)

def LowerCasing(sample):
    return sample.lower()

def StrippingHTML(sample):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', sample)

def NormalizingURL(sample):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'httpaddr', sample, flags=re.MULTILINE)

def NormalizingEmail(sample):
    return re.sub(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", 'emailaddr', sample, flags=re.MULTILINE)

def NormalizingNumber(sample):
    #return re.sub(r'[0-9\.]+', 'number', sample, flags=re.MULTILINE)
    return re.sub(r'\d*\.?\d+', 'number', sample, flags=re.MULTILINE)

def NormalizingDollar(sample):
    return re.sub(r'[$]+', ' dollar ', sample, flags=re.MULTILINE)

from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
def WordStemming(sample):
    stem1=LancasterStemmer()
    stem2=PorterStemmer()
    tokenWords=word_tokenize(sample)
    #tokenWords
    stem_sentence=[]
    for word in tokenWords:
        stem_sentence.append(stem1.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence), tokenWords  

def RemovalNonWord(sample):
    return re.sub(r'[^a-zA-Z ]', '', sample, flags=re.MULTILINE)

#Funcao para aplicar as diferentes funcoes de nornalizacao a um arquivo
#e retornar um arquivo nornalizado assim que a lista das palavras deste
#arquivo nornalizado
def NormalizacaoArquivo(sample):
    sample = LowerCasing(sample)
    sample = StrippingHTML(sample)
    sample = NormalizingURL(sample)
    sample = NormalizingEmail(sample)
    sample = NormalizingNumber(sample)
    sample = NormalizingDollar(sample)
    sample, tokenWords = WordStemming(sample)
    sample = RemovalNonWord(sample)

    return sample, tokenWords

dicCountWords = {}
#Informa o dicionario dicCountWords com o numero de ocorencia de cada palavra
#ficada nos arquivos lidos depois da nornalizacao 
def CountWords(dicCountWords, tokenWords):
    for word in tokenWords:
        if word in dicCountWords:
            dicCountWords[word] = dicCountWords[word]+1
        else:
            dicCountWords[word] = 1

''' === PARTE COMENTADA PARA DOWNLOAD DOS TAR DE EXEMPLOS DE SPAM-HAM ===        
import requests
url = 'http://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2'
if url.find('/'):
    filename = url.rsplit('/', 1)[1]
def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True
print(is_downloadable(url))

r = requests.get(url, allow_redirects=True)
print(r.headers.get('content-type'))
print(r.headers.get('content-length', None))
print(filename)
open(filename, 'wb').write(r.content)

url = 'http://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2'
if url.find('/'):
    filename = url.rsplit('/', 1)[1]
r = requests.get(url, allow_redirects=True)
print(filename)
open(filename, 'wb').write(r.content)

import zipfile
zfile = zipfile.ZipFile(filename)
for finfo in zfile.infolist():
    ifile = zfile.open(finfo)
    line_list = ifile.readlines()
    print(line_list)
    
======================= FIM PARTE COMENTADA '''     
import tarfile
#tfile = tarfile.TarFile(filename)
filename='20021010_spam.tar.bz2'
tf = tarfile.open(filename, 'r:bz2') #Arquivo tar com a compressao bzip2
#tf.list(verbose=True)
#tf.getnames()
#tf.getmembers()
listaNames = tf.getnames()
listaSamples = list()
for i in range(1, 2): #len(listaNames 
    name = listaNames[i]
    print('--> Iteracao {0}: Arquivo {1} a nornalizar'.format(i, name))
    file = tf.extractfile(name)
    sample = file.read().decode(errors='ignore')
    print(sample)
    sample, tokenWords = NormalizacaoArquivo(sample)
    CountWords(dicCountWords, tokenWords)
    listaSamples.append(sample) 
    
