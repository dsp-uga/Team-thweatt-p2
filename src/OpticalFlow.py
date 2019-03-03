#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:16:26 2019

@author: hemanth
"""

import cv2
import numpy as np
import os
import math
#https://waset.org/publications/10000781/color-image-segmentation-using-svm-pixel-classification-image

dir = "/Users/hemanth/Desktop/MSAI/DataSciencePracticum/Projects/p2/"
train_folder = dir+"pp_data_train"
test_folder = dir+"pp_data_test"
masks_folder = dir+"pp_masks"
pics = 30

def get_pp_images(train_folder,masks_folder,mask):
    folders = [f for f in os.listdir(train_folder) if not f.startswith('.')]
    X_train = []
    y_train = []
    for folder in folders:
        for file in os.listdir(train_folder+'/'+folder):
            X_train.append(np.load(train_folder+'/'+folder+'/'+file))
        if(mask == True):
            y_train.append(np.load(masks_folder+'/'+folder+'.npy'))
    if(mask == True):
        return X_train,y_train
    else:
        return X_train

def get_pp_images_test(test_folder):
    folders = [f for f in os.listdir(test_folder) if not f.startswith('.')]
    X_test = []
    y_test = []
    for folder in folders:
        for file in os.listdir(test_folder+'/'+folder):
            X_test.append(np.load(test_folder+'/'+folder+'/'+file))
        y_test.append(folder)
    return X_test,y_test



#Reading train and masks
if os.path.exists(train_folder) and os.path.exists(masks_folder):
    X_train,y_train = get_pp_images(train_folder,masks_folder,mask = True)
else:
    print('Pre-processed training/masks set not available')
    
#Reading test images
if os.path.exists(test_folder):
    X_test,test_names = get_pp_images_test(test_folder)
else:
    print('Pre-processed testing set not available')
    
ntrain = len([f for f in os.listdir(train_folder) if not f.startswith('.')])
ntest = len([f for f in os.listdir(test_folder) if not f.startswith('.')])


X_train = np.array(X_train)
X_train = np.reshape(X_train,(ntrain,pics))

y_train = np.array(y_train)

X_test = np.array(X_test)
X_test = np.reshape(X_test,(ntest,pics))


    
import pandas as pd
from sklearn.svm import SVC

def getFeaturesTrain(X_train):
    train_dict = {'angle' : [],'mag' : [],'mask' : []}
    train_df = pd.DataFrame(train_dict)
        
    #check for across 30 frames relative to one-another instead of relative to first img
    for j in range(len(X_train)):
        currentTrain = X_train[j]
        currentMask = y_train[j]
        
        prvs = currentTrain[0]
        
        for i in range(1,len(currentTrain)):
            next = currentTrain[i]
            flow = cv2.calcOpticalFlowFarneback(prvs, next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            angle = ang*180/np.pi/2
            magn = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            if(i == 1):
                mags = magn
                angs = angle
            else:
                mags = np.mean([mags,magn],axis = 0)
                angs = np.mean([angs,angle],axis = 0)
        
        angleC = angs.flatten()
        magnC = mags.flatten()
        maskC = currentMask.flatten()
        
        dict_temp = {'angle' : angleC,'mag' : magnC,'mask' : maskC}
        df = pd.DataFrame(dict_temp)
        df_2 = df[df['mask']==2]
        df_1 = df[df['mask']==1]
        df_0 = df[df['mask']==0]
        #cilia_count = (df_2.shape[0])/df.shape[0]
        #cell_count = (df_1.shape[0])/df.shape[0]
        #back_count = (df_0.shape[0])/df.shape[0]
        if(df_2.shape[0] != 0):
            df_cilia = df_2.sample(n = math.ceil(0.75*df_2.shape[0]))
        else:
            df_cilia = df_2
        if(df_1.shape[0] != 0):
            df_cell = df_1.sample(n = math.ceil(0.5*df_1.shape[0]))
        else:
            df_cell = df_1
        if(df_0.shape[0] != 0):
            df_back = df_0.sample(n = math.ceil(0.4*df_0.shape[0]))
        else:
            df_back = df_0
        train_df = pd.concat([train_df,df_cilia,df_cell,df_back])
    return(train_df)
    
def getFeaturesTest(X_test):
    test_dict = {'angle' : [],'mag' : []}
    test_df = pd.DataFrame(test_dict)
        
    currentTrain = X_test
    #currentMask = y_train[j]
    
    prvs = currentTrain[0]
    
    for i in range(1,len(currentTrain)):
        next = currentTrain[i]
        flow = cv2.calcOpticalFlowFarneback(prvs, next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        angle = ang*180/np.pi/2
        magn = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        if(i == 1):
            mags = magn
            angs = angle
        else:
            mags = np.mean([mags,magn],axis = 0)
            angs = np.mean([angs,angle],axis = 0)
    
    angleC = angs.flatten()
    magnC = mags.flatten()
    #maskC = currentMask.flatten()
    
    dict_temp = {'angle' : angleC,'mag' : magnC}
    
    df = pd.DataFrame(dict_temp)
    test_df = pd.concat([test_df,df])
    return(test_df)
    
train_DF = getFeaturesTrain(X_train)


train_DF['mask'] = train_DF['mask'].astype('category')




#df_2 = df[df['mask']==2]
#print(i)
#print("mag max : " +str(df_2['mag'].max()))
#print("mag min : " +str(df_2['mag'].min()))
#print("angle max : " +str(df_2['angle'].max()))
#print("angle min : " +str(df_2['angle'].min()))
#    
#    
#df_1 = df[df['mask']==1]
#df_0 = df[df['mask']==0]
#
#df.describe()
#df_2.describe()
#df_1.describe()
#df_0.describe()
#

from sklearn.ensemble import RandomForestClassifier


x = train_DF[['angle','mag']]
y = train_DF['mask']


clf = RandomForestClassifier(n_estimators=50, max_depth=7,random_state=0)
clf_fit = clf.fit(x, y) 

#clf = SVC(kernel = 'linear',C = 1.0)
#clf.fit(x, y) 
if not os.path.isdir(dir+"output"):
    os.mkdir(dir+"output")
    
import scipy.misc

for i in range(len(X_test)):
    currentTest = X_test[i]
    testName = test_names[i]
    dimX = currentTest[1].shape[0]
    dimY = currentTest[1].shape[1]
    test_DF = getFeaturesTest(currentTest)
    y_pred = clf.predict(test_DF[['angle','mag']])
    pred_img = np.reshape(y_pred,(dimX,dimY)).astype('uint8')
    scipy.misc.toimage(pred_img, cmin=0.0, cmax=...).save(dir+'output/'+testName+'.png')
    #cv2.imwrite(dir+'output/'+testName+'.png',pred_img)
    

#clf_fit.decision_path
#y_pred = clf.predict(test_DF[['angle','mag']])

#from sklearn.metrics import confusion_matrix
#confusion_matrix(y,y_pred)


#msk1 = cv2.imread(dir+"output/e297ad628b864c59a562c408d5caa1534cf0efc535235663741c5160f68bd4b5.png",0)
#msk1 = msk1.flatten()
#msk2 = cv2.imread(dir+"output/ccdd62ddc64f2e78cbd50a240fc337ef0ff781db6a89a1c5009dcc04cb903c2c.png",0)
#msk2 = msk2.flatten()
#msk = np.concatenate((msk1,msk2),axis = 0)
#
#confusion_matrix(msk,y_pred)
