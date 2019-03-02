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
    
    
(X_train[0].shape)    
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

##########################################
from PIL import Image 

img = X_train[0]
msk = y_train[0]
msk = msk.convert('RGB') 

cv2.imshow('img1',img)
cv2.waitKey()
cv2.destroyAllWindows()

msk[np.where(msk != 2)] = 0
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

import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# Read in the first frame
#frame1 = cv2.imread(X_train[0])
prvs = X_train[0]

#for i in range(1,len(X_train)):
next = X_train[1]
flow = cv2.calcOpticalFlowFarneback(prvs, next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
angle = ang*180/np.pi/2
magn = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
print(np.amax(magn))


angleC = angle.flatten()
magnC = magn.flatten()
maskC = msk.flatten()

dict_new = {            
'angle' : angleC,
'mag' : magnC,
'mask' : maskC
}


df = pd.DataFrame(dict_new)

df['mask'] = df['mask'].astype('category')
df_2 = df[df['mask']==255]
df_1 = df[df['mask']==1]
df_0 = df[df['mask']==0]

df.describe()
df_2.describe()
df_1.describe()
df_0.describe()


x = df[['angle','mag']]
y = df['mask']




clf =  DecisionTreeClassifier(random_state=0)
clf_fit = clf.fit(df[['angle','mag']],df['mask'])
clf_fit.decision_path
y_pred = clf.predict(x)

from sklearn.metrics import confusion_matrix
confusion_matrix(df['mask'],y_pred)

from sklearn.tree import _tree
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

tree_to_code(clf_fit,df.columns)



from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


dotfile = open("dt.dot", 'w')
tree.export_graphviz(dt, out_file=dotfile, feature_names=iris.feature_names)
dotfile.close()




# Set counter to read the second frame at the start
counter = 1

# Until we reach the end of the list...
while counter < len(X_train):
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