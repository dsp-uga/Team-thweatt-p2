import numpy as np
import tarfile
import cv2
import os
import requests
import pickle

#from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

pre_type = 'none'
pics = 30
dir = "/Users/hemanth/Desktop/MSAI/DataSciencePracticum/Projects/p2/"

#https://github.com/dsp-uga/goucher/blob/master/src/preprocessing/preprocessor.py
#https://github.com/dsp-uga/team-ball/blob/master/src/preprocessing/preprocessor.py

#small_lable = cv2.resize(mask, (mask.shape[1],mask.shape[0]),
 #                       interpolation=cv2.INTER_NEAREST)
#small_lable = (np.array(small_lable)).astype('uint8')
    
train_files = requests.get('https://storage.googleapis.com/uga-dsp/project2/train.txt').text.split()
test_files = requests.get('https://storage.googleapis.com/uga-dsp/project2/test.txt').text.split()

#Train files and masks pre-processing        
for names in train_files:
    img_file = tarfile.open(dir + "/data/" + names + ".tar")
    mask_file = dir + "/masks/" + names + ".png"
 
    #getting preprocessed mask images
    #load a color image in gray scale
    mask_imgs = cv2.imread(mask_file,0)
    if(pre_type == 'none'):
        mask_imgs = mask_imgs
    if not os.path.isdir(dir+"pp_masks"):
        os.mkdir(dir+"pp_masks")
    np.save(dir + "pp_masks"+'/' + names + ".npy", mask_imgs)
   
    #getting preprocessed train images
    
    extracted_img = img_file.getnames()[0:pics]
    if not os.path.isdir(dir+"pp_data_train"):
        os.mkdir(dir+"pp_data_train")
        
    os.mkdir(dir+"pp_data_train"+'/'+names)
    for n,img in enumerate(extracted_img):
        imgs = np.asarray(bytearray(img_file.extractfile(img).read()),dtype ="uint8")
        imgs = cv2.imdecode(imgs, cv2.IMREAD_GRAYSCALE)
        if(pre_type == 'none'):
            imgs = imgs
        #saving the intermediate arrays in a folder for future use
        np.save(dir + "pp_data_train"+'/' + names +'/img' + str(n) + ".npy", imgs)
        

#Test files pre-processing
for names in test_files:
    img_file = tarfile.open(dir + "/data/" + names + ".tar")
    extracted_img = img_file.getnames()[0:pics]
    if not os.path.isdir(dir+"pp_data_test"):
        os.mkdir(dir+"pp_data_test")
    
    os.mkdir(dir+"pp_data_test"+'/'+names)
    for n,img in enumerate(extracted_img):
        imgs = np.asarray(bytearray(img_file.extractfile(img).read()),dtype ="uint8")
        imgs = cv2.imdecode(imgs, cv2.IMREAD_GRAYSCALE)
        if(pre_type == 'none'):
            imgs = imgs
        #saving the intermediate arrays in a folder for future use
        np.save(dir + "pp_data_test"+'/' + names +'/img' + str(n) + ".npy", imgs)
    
