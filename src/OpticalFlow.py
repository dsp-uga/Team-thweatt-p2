#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:16:26 2019

@author: hemanth
"""

import cv2
import numpy as np

import os


dir = "/Users/hemanth/Desktop/MSAI/DataSciencePracticum/Projects/p2/"
train_folder = dir+"pp_optical"
test_folder = dir+"pp_data_test_zero_640x640"
masks_folder = dir+"pp_optical_masks"

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


#Reading train and masks
if os.path.exists(train_folder) and os.path.exists(masks_folder):
    X_train,y_train = get_pp_images(train_folder,masks_folder,mask = True)
else:
    print('Pre-processed training/masks set not available')
    
#Reading test images
#if os.path.exists(test_folder):
#    X_test = get_pp_images(test_folder,masks_folder,mask = False)
#else:
#    print('Pre-processed testing set not available')
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

##########################################

img = X_train[9]
msk = y_train[9]

cv2.imshow('img1',img)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('msk',msk)
cv2.waitKey()
cv2.destroyAllWindows()


np.set_printoptions(threshold=np.nan)


np.unique(img)
np.unique(msk)

np.where(X_train[2] > 2)
img[np.where(img == 2)] = 256
img[np.where(img == 1)] = 100

backtorgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


##########################################





# Create list of names here from my1.bmp up to my20.bmp
list_names = ['my' + str(i+1) + '.bmp' for i in range(20)]

# Read in the first frame
frame1 = cv2.imread(list_names[0])
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# Set counter to read the second frame at the start
counter = 1

# Until we reach the end of the list...
while counter < len(list_names):
    # Read the next frame in
    frame2 = cv2.imread(list_names[counter])
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Calculate optical flow between the two frames
    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Normalize horizontal and vertical components
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')

    # Show the components as images
    cv2.imshow('Horizontal Component', horz)
    cv2.imshow('Vertical Component', vert)

    # Change - Make next frame previous frame
    prvs = next.copy()

    # If we get to the end of the list, simply wait indefinitely
    # for the user to push something
    if counter == len(list_names)-1
        k = cv2.waitKey(0) & 0xff
    else: # Else, wait for 1 second for a key
        k = cv2.waitKey(1000) & 0xff

    if k == 27:
        break
    elif k == ord('s'): # Change
        cv2.imwrite('opticalflow_horz' + str(counter) + '-' + str(counter+1) + '.pgm', horz)
        cv2.imwrite('opticalflow_vert' + str(counter) + '-' + str(counter+1) + '.pgm', vert)

    # Increment counter to go to next frame
    counter += 1

cv2.destroyAllWindows()