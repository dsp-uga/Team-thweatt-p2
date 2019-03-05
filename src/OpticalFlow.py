import cv2
import numpy as np
import os
import math
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import scipy.misc
import time
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

dir = "/Users/hemanth/Desktop/MSAI/DataSciencePracticum/Projects/p2/"
train_folder = dir+"pp_data_train"
test_folder = dir+"pp_data_test"
masks_folder = dir+"pp_masks"
pics = 30

def get_variance(imgs):
    imgs_shape = np.stack(imgs)
    var_imgs = np.var(imgs_shape,axis = 0)
    var_flat = var_imgs.flatten()
    var_flat_scaled = np.interp(var_flat, (var_flat.min(), var_flat.max()), (0,1))
    return var_flat_scaled


def get_pp_images(train_folder,masks_folder):
    folders = [f for f in os.listdir(train_folder) if not f.startswith('.')]
    X_train = []
    y_train = []
    for folder in folders:
        for file in os.listdir(train_folder+'/'+folder):
            X_train.append(np.load(train_folder+'/'+folder+'/'+file))
        mask = np.load(masks_folder+'/'+folder+'.npy')
        mask[np.where(mask != 2)] = 0
        y_train.append(mask)
    return X_train,y_train


def get_pp_images_test(test_folder):
    folders = [f for f in os.listdir(test_folder) if not f.startswith('.')]
    X_test = []
    y_test = []
    for folder in folders:
        for file in os.listdir(test_folder+'/'+folder):
            X_test.append(np.load(test_folder+'/'+folder+'/'+file))
        y_test.append(folder)
    return X_test,y_test

def get_features_train(X_train):
	#Creating empty dataframe to populate with train features
    train_dict = {'angle' : [],'mag' : [],'intnC': [],'fftC':[],'varC': [],'mask' : []}
    train_df = pd.DataFrame(train_dict)
    
    #Finding Optical flow for all frames related to the first frame and FFT
    for j in range(len(X_train)):
        currentTrain = X_train[j]
        currentMask = y_train[j]
        
        prvs = currentTrain[0]
        
        for i in range(1,len(currentTrain)):
            next = currentTrain[i]
            #Calculating optical flow
            flow = cv2.calcOpticalFlowFarneback(prvs, next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            angle = ang*180/np.pi/2
            magn = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            
            #Calculating FFT
            dft = cv2.dft(np.float32(next),flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
            magnitude_spectrum_n = cv2.normalize(magnitude_spectrum,None,0,255,cv2.NORM_MINMAX)

            if(i == 1):
                mags = magn
                angs = angle
                ffts = magnitude_spectrum_n
            else:
                mags = np.sum([mags,magn],axis = 0)
                angs = np.mean([angs,angle],axis = 0)
                ffts = np.mean([ffts,magnitude_spectrum_n],axis = 0)

		#Flattening all array to form a single dataframe        
        angleC = angs.flatten()
        magnC = mags.flatten()
        intnC = prvs.flatten()
        fftC = ffts.flatten()
        varC = get_variance(currentTrain)
        maskC = currentMask.flatten()
        
        #Creating temporary dataframe for each hash
        dict_temp = {'angle' : angleC,'mag' : magnC,'intnC': intnC,'fftC': fftC,'varC': varC,'mask' : maskC}
        df = pd.DataFrame(dict_temp)
        
        #Equal rate sampling
        df_2 = df[df['mask']==2]
        df_0 = df[df['mask']==0]
        if(df_2.shape[0] != 0):
            df_cilia = df_2.sample(n = math.ceil(0.75*df_2.shape[0]))
        else:
            df_cilia = df_2
        if(df_0.shape[0] != 0):
            df_back = df_0.sample(n = math.ceil(0.25*df_0.shape[0]))
        else:
            df_back = df_0
        train_df = pd.concat([train_df,df_cilia,df_back])
    return(train_df)
    
def get_features_test(X_test): 
    currentTrain = X_test
    prvs = currentTrain[0]
    
    for i in range(1,len(currentTrain)):
        next = currentTrain[i]
        flow = cv2.calcOpticalFlowFarneback(prvs, next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        angle = ang*180/np.pi/2
        magn = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        
        dft = cv2.dft(np.float32(next),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
        magnitude_spectrum_n = cv2.normalize(magnitude_spectrum,None,0,255,cv2.NORM_MINMAX)

        if(i == 1):
            mags = magn
            angs = angle
            ffts = magnitude_spectrum_n
        else:
            mags = np.sum([mags,magn],axis = 0)
            angs = np.mean([angs,angle],axis = 0)
            ffts = np.mean([ffts,magnitude_spectrum_n],axis = 0)
    
    angleC = angs.flatten()
    magnC = mags.flatten()
    intnC = prvs.flatten()
    fftC = ffts.flatten()
    varC = get_variance(currentTrain)
        
    dict_temp = {'angle' : angleC,'mag' : magnC,'intnC': intnC,'fftC': fftC,'varC': varC}
    
    test_df = pd.DataFrame(dict_temp)
    return(test_df)

#Checking if the preprocessed train,test, and masks folders are generated in the project directory
if not os.path.isdir(train_folder):
	sys.exit("Pre-Processed train folder is missing in project directory, run the pre-process script to generate the folder")
if not os.path.isdir(masks_folder):
    sys.exit("Pre-Processed mask folder is missing in project directory, run the pre-process script to generate the folder")
if not os.path.isdir(test_folder):
    sys.exit("Pre-Processed test folder is missing in project directory, run the pre-process script to generate the folder")

#Reading train and masks
if os.path.exists(train_folder) and os.path.exists(masks_folder):
    X_train,y_train = get_pp_images(train_folder,masks_folder)
else:
    print('Pre-processed training/masks set not available')
    
#Reading test images and their hashes
if os.path.exists(test_folder):
    X_test,test_names = get_pp_images_test(test_folder)
else:
    print('Pre-processed testing set not available')

#Finding the number of training and testing hashes
ntrain = len([f for f in os.listdir(train_folder) if not f.startswith('.')])
ntest = len([f for f in os.listdir(test_folder) if not f.startswith('.')])

#Converting X_train,y_train, and X_test to single numpy array and reshaping
X_train = np.array(X_train)
X_train = np.reshape(X_train,(ntrain,pics))

y_train = np.array(y_train)

X_test = np.array(X_test)
X_test = np.reshape(X_test,(ntest,pics))

#Finding features for training set
train_DF = get_features_train(X_train)

#Changing mask column to type category
train_DF['mask'] = train_DF['mask'].astype('category')

#Selecting train features
x = train_DF[['angle','mag','intnC','fftC','varC']]
y = train_DF['mask']

#Training RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0,verbose = 1,n_jobs = -1)
clf_fit = clf.fit(x, y) 

#Creating output directory to write predictions
if not os.path.isdir(dir+"output"):
    os.mkdir(dir+"output")

#Predicting for each test image and writing to output directory
for i in range(len(X_test)):
    print("Predicting on test sample "+str(i+1))
    currentTest = X_test[i]
    testName = test_names[i]
    dimX = currentTest[1].shape[0]
    dimY = currentTest[1].shape[1]
    test_DF = getFeaturesTest(currentTest)
    y_pred = clf_fit.predict_proba(test_DF[['angle','mag','intnC','fftC','varC']])
    y_pred_cut = np.where(y_pred[:,1]>0.40, 2,0)
    pred_img = np.reshape(y_pred_cut,(dimX,dimY)).astype('uint8')
    scipy.misc.toimage(pred_img, cmin=0.0, cmax=...).save(dir+'output/'+testName+'.png')

