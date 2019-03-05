import numpy as np
import cv2
import os
import requests
import pickle
from skimage.transform import resize
import sys

pre_type = 'none'
dimX = 640
dimY = 640
pics = 30
dir = "/Users/hemanth/Desktop/MSAI/DataSciencePracticum/Projects/p2/"

def zero_padding(input_array, final_dimension = [dimX,dimY]):
    input_array_size = input_array.shape
    temp = np.zeros((final_dimension[0],final_dimension[1]))
    temp[:input_array_size[0],:input_array_size[1]] = input_array
    temp = np.expand_dims(temp, axis=-1)
    return temp
 
#Getting list of train and test files from "uga-dsp" google bucket
train_files = requests.get('https://storage.googleapis.com/uga-dsp/project2/train.txt').text.split()
test_files = requests.get('https://storage.googleapis.com/uga-dsp/project2/test.txt').text.split()

#Checking if the data and masks directories are present in the project directory
if not os.path.isdir(dir+"/data"):
	sys.exit("data folder is missing in project directory, download the folder from uga-dsp bucket and run again")
	
if not os.path.isdir(dir+"/masks"):
    sys.exit("masks folder is missing in project directory, download the folder from uga-dsp bucket and run again")

#Unzipping Train files, Masks, and also Pre-Processing        
for names in train_files:
    img_file = tarfile.open(dir + "/data/" + names + ".tar")
    mask_file = dir + "/masks/" + names + ".png"
 
    #Preprocessing Masks
    mask_imgs = cv2.imread(mask_file,0)
    if(pre_type == 'none'):
        mask_imgs = mask_imgs
    elif(pre_type == 'zero'):
        mask_imgs = zero_padding(mask_imgs)
    elif(pre_type == 'resize'):
        mask_imgs = np.array(cv2.resize(mask_imgs, (dimX, dimY),interpolation=cv2.INTER_NEAREST)).astype('uint8')
        mask_imgs = np.expand_dims(mask_imgs, axis=-1)
   	#Writing pre-processed images to new folder 'pp_masks'
    if not os.path.isdir(dir+"pp_masks"):
        os.mkdir(dir+"pp_masks")
    np.save(dir + "pp_masks"+'/' + names + ".npy", mask_imgs)
   
    #Preprocessing Train images   
    extracted_img = img_file.getnames()[0:pics]
    
    if not os.path.isdir(dir+"pp_data_train"):
        os.mkdir(dir+"pp_data_train")
    
    #Creating directory for each individual training hash
    if not os.path.isdir(dir+"pp_data_train"):    
    	os.mkdir(dir+"pp_data_train"+'/'+names)
    
    for n,img in enumerate(extracted_img):
        imgs = np.asarray(bytearray(img_file.extractfile(img).read()),dtype ="uint8")
        imgs = cv2.imdecode(imgs, cv2.IMREAD_GRAYSCALE)
        if(pre_type == 'none'):
            imgs = imgs
        elif(pre_type == 'zero'):
            imgs = zero_padding(imgs)
        elif(pre_type == 'resize'):
            imgs = np.array(cv2.resize(imgs, (dimX,dimY),interpolation=cv2.INTER_NEAREST)).astype('uint8')
            imgs = np.expand_dims(imgs, axis=-1)

        #Writing pre-processed train images to new folder 'pp_data_train'
        np.save(dir + "pp_data_train"+'/' + names +'/img' + str(n) + ".npy", imgs)


#Test files pre-processing and also populating list of original image sizes for each test hash
for names in test_files:
    img_file = tarfile.open(dir + "/data/" + names + ".tar")
    extracted_img = img_file.getnames()[0:pics]
    
    #Creating directory to save test hashes
    if not os.path.isdir(dir+"pp_data_test"):
        os.mkdir(dir+"pp_data_test")
    
    #Creating directory for each individual training hash
    if not os.path.isdir(dir+"pp_data_test"):    
    	os.mkdir(dir+"pp_data_test"+'/'+names)
    
    for n,img in enumerate(extracted_img):
        imgs = np.asarray(bytearray(img_file.extractfile(img).read()),dtype ="uint8")
        imgs = cv2.imdecode(imgs, cv2.IMREAD_GRAYSCALE)
        
        #Appending original dimensions of image to list
        original_size = imgs.shape
        test_dimensions.append((names,original_size))        
        
        if(pre_type == 'none'):
            imgs = imgs
        elif(pre_type == 'zero'):
            imgs = zero_padding(imgs)
        elif(pre_type == 'resize'):
            imgs = np.array(cv2.resize(imgs, (dimX,dimY),interpolation=cv2.INTER_NEAREST)).astype('uint8')
            imgs = np.expand_dims(imgs, axis=-1)
 
        #Writing pre-processed test images to new folder 'pp_data_test'
        np.save(dir + "pp_data_test"+'/' + names +'/img' + str(n) + ".npy", imgs)
    
#Saving populated list of test hashes with original dimensions
with open(dir+'test_dimensions.pkl', 'wb') as f:
    pickle.dump(test_dimensions, f)
