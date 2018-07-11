#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:29:59 2018

@author: anikur93
"""

import os
import numpy as np
import matplotlib.image as mpimg
import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import StratifiedKFold
#from alexnet_conv import AlexNet
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import operator
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras import optimizers

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.layers import Activation, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
path = r'refined_images'
#path = r''

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    images = []
    labels = []
    for image_path in image_paths:
        image = load_img(image_path)
        label = os.path.split(image_path)[1].split('.')[0].split('_')[0]
        #label = ''.join(x for x in nbr if x.isalpha())
        images.append(image)
        labels.append(label)
        #cv2.imshow('Adding faces to training set...', image)
        #cv2.waitKey(50)
    return images, labels

images, labels = get_images_and_labels(path)


#def create_pairs(X_train, y_train):
#    tr_pairs = []
#    tr_y = []
#
#    y_train = np.array(y_train)
#    digit_indices = [np.where(y_train == i)[0] for i in list(set(y_train))]
#    
#    for i in range(len(digit_indices)):
#        n = len(digit_indices[i])
#        for j in range(n):
#            random_index = digit_indices[i][j]
#            anchor_image = X_train[random_index]
#            anchor_label = y_train[random_index]
#            anchor_indices = [i for i, x in enumerate(y_train) if x == anchor_label]
#            negate_indices = list(set(list(range(0,len(X_train)))) - set(anchor_indices))
#            for k in range(j+1,n):
#                support_index = digit_indices[i][k]
#                support_image = X_train[support_index]
#                tr_pairs += [[anchor_image,support_image]]
#                #print(k)
#                negate_index = random.choice(negate_indices)
#                negate_image = X_train[negate_index]
#                tr_pairs += [[anchor_image,negate_image]]
#                #print(k)
#                tr_y += [1,0]
#                #print(k)
#            
#    return np.array(tr_pairs),np.array(tr_y) 

#def checking(n):
#    plt.imshow(X_train[n])
#    print(y_train[n])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def normalise(list1):
    l = []
    for i in range(len(list1)):
        img = list1[i]
        norm = (img - 128)/128
        l.append(norm)
    return l


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
#    seq = Sequential()
#    model.add(Conv2D(32, (3, 3), padding='same',
#                 input_shape=(220,220,1)))
#    model.add(Activation('relu'))
#    model.add(Conv2D(32, (3, 3)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#    
#    model.add(Conv2D(64, (3, 3), padding='same'))
#    model.add(Activation('relu'))
#    model.add(Conv2D(64, (3, 3)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
#    model.add(Flatten())
#    model.add(Dense(512, input_shape=(48400,),activation = 'tanh'))
##    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(256))
#    model.add(Activation('tanh'))
#    model.add(Dropout(0.5))
#    model.add(Dense(128))
#    model.add(Activation('tanh'))
#    model.add(Dropout(0.5))
#    model.add(Dense(64))
#    model.add(Activation('tanh'))
#    model.add(Dropout(0.5))
#    model.add(Activation('softmax'))
    
#    seq.add(Dense(128, input_shape=(input_dim,), activation='tanh'))
#    seq.add(Dense(128, input_shape=(input_dim,), activation='tanh'))
#    seq.add(Dense(128, input_shape=(input_dim,), activation='tanh'))
#    seq.add(Dense(64, activation='tanh'))
    seq = Sequential()
    seq.add(Dense(1024, input_shape=(input_dim,), activation='tanh'))
#    seq.add(Dropout(0.5))
#    seq.add(Dense(1024, activation='tanh'))
    seq.add(Dense(512, activation='tanh'))
    seq.add(Dense(256, activation='tanh'))
    seq.add(Dense(128, activation='tanh'))
#    seq.add(Dropout(0.2))
#    seq.add(Dense(64, activation='sigmoid'))
    return seq
#    return model


#def compute_accuracy(predictions, labels):
#    '''Compute classification accuracy with a fixed threshold on distances.
#    '''
#    return labels[predictions.ravel() < 0.6].mean()
def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return 1-(abs(labels-(predictions.ravel()<0.5)).sum()/len(labels))


def create_allpairs(X_train, y_train):
    tr_pairs = []
    tr_y = []

    y_train = np.array(y_train)
    digit_indices = [np.where(y_train == i)[0] for i in list(set(y_train))]
    pair_sets = []
    for i in range(len(digit_indices)):
        n = len(digit_indices[i])
        for j in range(n):
            random_index = digit_indices[i][j]
            anchor_image = X_train[random_index]
            anchor_label = y_train[random_index]
            anchor_indices = [i for i, x in enumerate(y_train) if x == anchor_label]
            negate_indices = list(set(list(range(0,len(X_train)))) - set(anchor_indices))
            for k in range(j+1,n):
                support_index = digit_indices[i][k]
                support_image = X_train[support_index]
                tr_pairs += [[anchor_image,support_image]]
                #print(k)
                #negate_index = random.choice(negate_indices)
                #negate_image = X_train[negate_index]
                #tr_pairs += [[anchor_image,negate_image]]
                #print(k)
                tr_y += [1]
                #print(k)
            for m in range(len(negate_indices)-270):
                negate_index = random.choice(negate_indices)
                se = set([random_index, negate_index])
                if se not in pair_sets:
                    negate_image = X_train[negate_index]
                    tr_pairs += [[anchor_image,negate_image]]
                    tr_y += [0]
                    pair_sets.extend([se])
                
           
    return np.array(tr_pairs),np.array(tr_y) 

def create_allpairs_test(X_train, y_train):
    tr_pairs = []
    tr_y = []

    y_train = np.array(y_train)
    digit_indices = [np.where(y_train == i)[0] for i in list(set(y_train))]
    pair_sets = []
    for i in range(len(digit_indices)):
        n = len(digit_indices[i])
        for j in range(n):
            random_index = digit_indices[i][j]
            anchor_image = X_train[random_index]
            anchor_label = y_train[random_index]
            anchor_indices = [i for i, x in enumerate(y_train) if x == anchor_label]
            negate_indices = list(set(list(range(0,len(X_train)))) - set(anchor_indices))
            for k in range(j+1,n):
                support_index = digit_indices[i][k]
                support_image = X_train[support_index]
                tr_pairs += [[anchor_image,support_image]]
                #print(k)
                #negate_index = random.choice(negate_indices)
                #negate_image = X_train[negate_index]
                #tr_pairs += [[anchor_image,negate_image]]
                #print(k)
                tr_y += [1]
                #print(k)
            for m in range(len(negate_indices)):
                negate_index = random.choice(negate_indices)
                se = set([random_index, negate_index])
                if se not in pair_sets:
                    negate_image = X_train[negate_index]
                    tr_pairs += [[anchor_image,negate_image]]
                    tr_y += [0]
                    pair_sets.extend([se])
                    
    return np.array(tr_pairs),np.array(tr_y) 


X_train = []
y_train = labels

for i in range(len(images)):
    img = images[i].resize((220, 220))
    X_train.append(img)
    
for i in range(len(X_train)):
    X_train[i] = np.array(X_train[i])


###WE ALREADY HAVE NUMBERED LABELS AND SO NO LABELENCODING
#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#X_train = []
#
#le.fit(y_train)
#le.classes_
#y_train = le.transform(y_train)
#le.inverse_transform(y_train)

#le.transform(le.classes_)


#label_sets = list(set(labels))
#y_train = labels

#nb_classes = len(label_sets)

X_train, y_train = shuffle(X_train, y_train)



#X_test = X_val
#y_test = y_val
X_train = np.array(X_train).astype('float32')
#X_test = np.array(X_test).astype('float32')
#y_train = np.array(y_train).astype('float32')
#y_test = np.array(y_test).astype('float32')

X_train = rgb2gray(X_train)
#X_test = rgb2gray(X_test)

#x_gray = conversion(X_train)
#x_nor_gray = normalise(x_gray) 

#with open('./train.p', 'rb') as f:
#    data = pickle.load(f)
x_train = normalise(X_train)
X_train = x_train
#X_train = np.array(X_train).reshape(383,220,220,1)
X_train = np.array(X_train).reshape(383,48400)


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=2)
#X_test, X_test2, y_test, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=2)


X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')

tr_pairs, tr_y = create_allpairs(X_train, y_train)            
#te_pairs, te_y = create_allpairs_test(X_test, y_test)

#input_dim = (220,220,1)
input_dim = 48400
nb_epoch = 7

y_train = np.array(y_train)
y_test = np.array(y_test)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

cvscores = []
for train, test in kfold.split(tr_pairs, tr_y):
    base_network = create_base_network(input_dim)
    
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    model = Model(input=[input_a, input_b], output=distance)
    
    
    rms = optimizers.SGD()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[train][:, 0], tr_pairs[train][:, 1]], tr_y[train],
              batch_size=128,verbose=1,
    nb_epoch=nb_epoch)
    scores = model.evaluate([tr_pairs[test][:, 0], tr_pairs[test][:, 1]], tr_y[test], verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

for train, test in kfold.split(tr_pairs, tr_y):
    print(len(tr_pairs[train]),(tr_y[train][0:20]),(tr_y[test][0:20]))

pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

def create_predict_pairs(temp, X_train, y_train):
    tr_pairs = []
    tr_y = []
    #X_train = np.array(X_train)
    y_train = np.array(y_train)
    for i in range(len(X_train)):
        tr_pairs += [[temp,X_train[i]]]
        tr_y += [y_train[i]] 
    return np.array(tr_pairs),np.array(tr_y) 

  
def predict_class(predi,labels):
    l = []
    for i in range(len(predi)):
        if predi[i] == True:
            l.extend([labels[i]])
    return l


def test_accu_point(index, X_train,y_train,X_test, y_test, threshold):
    temp = X_test[index]
    pairs, labels = create_predict_pairs(temp,X_train, y_train)
    predicti = model.predict([pairs[:,0],pairs[:,1]])
    predi = predicti < threshold
    l = predict_class(predi, labels)
    count = Counter(l)
    if len(count.most_common()) != 0:
        if count.most_common()[0][0] == y_test[index]:
            return 'Matched'
            #print(count.most_common()[0][0],y_test[index])
        else :
            return 'No match'
            #print(count.most_common()[0][0],y_test[index])
    else:
        return 'No match'
    
def real_accuracy_maj(X_test):
    s = 0
    for i in range(len(X_test)):
        output = test_accu_point(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test,threshold=0.4)
        if output == 'Matched':
            s += 1
    print(s,len(X_test), 100*s/len(X_test))
    
    
def test_closet(index,X_train,y_train,X_test, y_test):
    temp = X_test[index]
    pairs, labels = create_predict_pairs(temp,X_train, y_train)
    predicti = model.predict([pairs[:,0],pairs[:,1]])
    min_index = np.argmin(predicti)
    prec_label = labels[min_index]
    if prec_label == y_test[index]:
        return 'Matched'
    else:
        return 'No match'
    
def real_accuracy_oneclose(X_test):
    s = 0
    for i in range(len(X_test)):
        output = test_closet(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test)
        if output == 'Matched':
            s += 1
    print(s,len(X_test), 100*s/len(X_test))
   
def test_closet_n(index,X_train,y_train,X_test, y_test,n):
    temp = X_test[index]
    pairs, labels = create_predict_pairs(temp,X_train, y_train)
    predicti = model.predict([pairs[:,0],pairs[:,1]])
    new_pre = []
    for k in range(len(predicti)):
        new_pre.extend([predicti[k][0]])
    min_indexes = np.array(new_pre).argsort()[:n] 
    prec_labels =[]
    for j in range(n):
        prec_labels.extend([labels[min_indexes[j]]])  
    #print(prec_labels)
    count = Counter(prec_labels)
    prec_label = count.most_common()[0][0]
    if prec_label == y_test[index]:
        return 'Matched'
    else:
        return 'No match'
    
def real_accuracy_n_close(X_test):
    s = 0
    for i in range(len(X_test)):
        output = test_closet_n(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test, n=3)
        if output == 'Matched':
            s += 1
    print(s,len(X_test), 100*s/len(X_test))
    
    
def real_all3():
    s = 0
    for i in range(len(X_test)):
        output1 = test_accu_point(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test,threshold=0.4)
        output2 = test_closet(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test)
        output3 = test_closet_n(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test, n=3)
        if (output1 == 'Matched') | (output2 == 'Matched') | (output3 == 'Matched'):
            s += 1
    print(s,len(X_test), 100*s/len(X_test))
                
    
def real_voting():
    s = 0
    for i in range(len(X_test)):
        output1 = test_accu_point(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test,threshold=0.4)
        output2 = test_closet(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test)
        output3 = test_closet_n(index=i, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test, n=3)
        if ((output1 == 'Matched') & (output2 == 'Matched')) | ((output1 == 'Matched') & (output3 == 'Matched')) | ((output3 == 'Matched') & (output2 == 'Matched')) :
            s += 1
    print(s,len(X_test), 100*s/len(X_test))
                
            
            
real_accuracy_maj(X_test=X_test)##40.25
real_accuracy_oneclose(X_test=X_test)##58.44
real_accuracy_n_close(X_test=X_test)##57.14
real_all3()##63.63
real_voting()##59.74


real_accuracy_maj(X_test=X_test2)##40.25
real_accuracy_oneclose(X_test=X_test2)##58.44
real_accuracy_n_close(X_test=X_test2)##57.14
real_all3()##63.63
real_voting()##59.74
    
test_closet(index=32, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test)
    

test_accu_point(index=32, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test,threshold=0.4)

index = random.randint(0,len(X_test))
#index = 32
temp = X_test[index]


pairs, labels = create_predict_pairs(temp,X_train, y_train)
predicti = model.predict([pairs[:,0],pairs[:,1]])
predi = predicti < 0.3
  
l = predict_class(predi, labels)


from collections import Counter
count = Counter(l)
count    
print(count.most_common())
y_test[index]
    
#pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
#tr_acc = compute_accuracy(pred, tr_y)
#new_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc_new, real_labels = real_accuracy(X_test, X_train, y_test, y_train, threshold=0.4)

#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Real Accuracy on test set: %0.2f%%' % (100 * te_acc_new))
    
