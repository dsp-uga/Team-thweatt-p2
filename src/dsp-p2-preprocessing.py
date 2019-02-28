import numpy as np
import tarfile
import cv2
import os
import requests
import pickle

#from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

pre_type = 'resize'
dimX = "512"
dimY = "512"
pics = '1'
dir = "/Users/hemanth/Desktop/MSAI/DataSciencePracticum/Projects/p2/"

#https://github.com/dsp-uga/goucher/blob/master/src/preprocessing/preprocessor.py
#https://github.com/dsp-uga/team-ball/blob/master/src/preprocessing/preprocessor.py

#small_lable = cv2.resize(mask, (mask.shape[1],mask.shape[0]),
 #                       interpolation=cv2.INTER_NEAREST)
#small_lable = (np.array(small_lable)).astype('uint8')


def zero_padding(input_array, final_dimension = [int(dimX),int(dimY)]):
    input_array_size = input_array.shape
    temp = np.zeros((final_dimension[0],final_dimension[1]))
    temp[:input_array_size[0],:input_array_size[1]] = input_array
    temp = np.expand_dims(temp, axis=-1)
    return temp
    
train_files = requests.get('https://storage.googleapis.com/uga-dsp/project2/train.txt').text.split()
test_files = requests.get('https://storage.googleapis.com/uga-dsp/project2/test.txt').text.split()

#Train files and masks pre-processing        
for names in train_files:
    img_file = tarfile.open(dir + "/data/" + names + ".tar")
    mask_file = dir + "/masks/" + names + ".png"
 
    #getting preprocessed mask images
    #load a color image in gray scale
    mask_imgs = cv2.imread(mask_file,0)
    if(pre_type == 'zero'):
        mask_imgs = zero_padding(mask_imgs)
    elif(pre_type == 'resize'):
        mask_imgs = np.array(cv2.resize(mask_imgs, (int(dimX), int(dimY)),interpolation=cv2.INTER_NEAREST)).astype('uint8')
        mask_imgs = np.expand_dims(mask_imgs, axis=-1)
    
    if not os.path.isdir(dir+"pp_masks_"+pre_type+"_"+dimX+"x"+dimY):
        os.mkdir(dir+"pp_masks_"+pre_type+"_"+dimX+"x"+dimY)
    np.save(dir + "pp_masks_"+pre_type+"_"+dimX+"x"+dimY+'/' + names + ".npy", mask_imgs)
   
    #getting preprocessed train images
    
    extracted_img = img_file.getnames()[0:int(pics)]
    if not os.path.isdir(dir+"pp_data_train_"+pre_type+"_"+dimX+"x"+dimY):
        os.mkdir(dir+"pp_data_train_"+pre_type+"_"+dimX+"x"+dimY)
    os.mkdir(dir+"pp_data_train_"+pre_type+"_"+dimX+"x"+dimY+'/'+names)
    for n,img in enumerate(extracted_img):
        imgs = np.asarray(bytearray(img_file.extractfile(img).read()),dtype ="uint8")
        imgs = cv2.imdecode(imgs, cv2.IMREAD_GRAYSCALE)
        if(pre_type == 'zero'):
            imgs = zero_padding(imgs)
        elif(pre_type == 'resize'):
            imgs = np.array(cv2.resize(imgs, (int(dimX), int(dimY)),interpolation=cv2.INTER_NEAREST)).astype('uint8')
            imgs = np.expand_dims(imgs, axis=-1)

        #saving the intermediate arrays in a folder for future use
        np.save(dir + "pp_data_train_"+pre_type+"_"+dimX+"x"+dimY+'/' + names +'/img' + str(n) + ".npy", imgs)
        

#Test files pre-processing
test_dimensions = []
for names in test_files:
    img_file = tarfile.open(dir + "/data/" + names + ".tar")
    extracted_img = img_file.getnames()[0:int(pics)]
    if not os.path.isdir(dir+"pp_data_test_"+pre_type+"_"+dimX+"x"+dimY):
        os.mkdir(dir+"pp_data_test_"+pre_type+"_"+dimX+"x"+dimY)
    
    os.mkdir(dir+"pp_data_test_"+pre_type+"_"+dimX+"x"+dimY+'/'+names)
    for n,img in enumerate(extracted_img):
        imgs = np.asarray(bytearray(img_file.extractfile(img).read()),dtype ="uint8")
        imgs = cv2.imdecode(imgs, cv2.IMREAD_GRAYSCALE)
        original_size = imgs.shape
        test_dimensions.append((names,original_size))        
        if(pre_type == 'zero'):
            imgs = zero_padding(imgs)
        elif(pre_type == 'resize'):
            imgs = np.array(cv2.resize(imgs, (int(dimX), int(dimY)),interpolation=cv2.INTER_NEAREST)).astype('uint8')
            imgs = np.expand_dims(imgs, axis=-1)

        #saving the intermediate arrays in a folder for future use
        np.save(dir + "pp_data_test_"+pre_type+"_"+dimX+"x"+dimY+'/' + names +'/img' + str(n) + ".npy", imgs)
    
with open(dir+'test_dimensions.pkl', 'wb') as f:
    pickle.dump(test_dimensions, f)
