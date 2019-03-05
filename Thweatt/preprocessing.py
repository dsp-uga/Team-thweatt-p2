import numpy as np
import requests
import tarfile
import pickle
import sys
import cv2
import os

from util import zero_padding

def load_train(train_file, base_dir, dim_x=640, dim_y=640, n_frames=30):
	"""A funciton to unzip, preprocess, and store training image files as numpy arrays.
	
	:param train_file: a file containing training hashes
	:param base_dir: a directory where processed training images are saved
	:param dim_x: an integer identifying the width of the image
	:param dim_y: an integer identifying the length of the image
	:param n_frames: an integer specifying the number of frames per video
	:returns: None
	:raises NotADirectoryError: if /data and /masks cannot be located inside base_dir directory
	"""

	# checking if the train_file exist, if not download it from GC bucket
	if os.path.isfile(train_file):
		train_files = open(train_file).read().split('\n')
	else:
		train_files = requests.get('https://storage.googleapis.com/uga-dsp/project2/train.txt').text.split()


	# Checking if the data and masks directories are present in the project directory
	if not os.path.isdir(base_dir + "/data"):
		raise NotADirectoryError("data folder is missing in project directory, download the folder from uga-dsp bucket and run again")
	
	if not os.path.isdir(base_dir + "/masks"):
		raise NotADirectoryError("masks folder is missing in project directory, download the folder from uga-dsp bucket and run again")

	for file_name in train_files:
		img_file = tarfile.open(base_dir + "/data/" + file_name + ".tar")
		mask_file = base_dir + "/masks/" + file_name + ".png"
	 
		# Preprocessing Masks
		mask_imgs = cv2.imread(mask_file,0)
		if pre_type == 'none':
			mask_imgs = mask_imgs
		elif pre_type == 'zero':
			mask_imgs = zero_padding(mask_imgs)
		elif pre_type == 'resize':
			mask_imgs = np.array(cv2.resize(mask_imgs, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)).astype('uint8')
			mask_imgs = np.expand_dims(mask_imgs, axis=-1)

	   	# Writing pre-processed images to new folder 'pp_masks'
		if not os.path.isdir(base_dir + "pp_masks"):
			os.mkdir(base_dir + "pp_masks")
		np.save(base_dir + "pp_masks/" + file_name + ".npy", mask_imgs)
	   
		# Preprocessing Train images   
		extracted_img = img_file.getnames()[0:n_frames]
		
		if not os.path.isdir(base_dir + "pp_data_train"):
			os.mkdir(base_dir + "pp_data_train")
		
		# Creating directory for each individual training hash
		if not os.path.isdir(base_dir + "pp_data_train"):	
			os.mkdir(base_dir + "pp_data_train/" + file_name)
		
		for n, img in enumerate(extracted_img):
			imgs = np.asarray(bytearray(img_file.extractfile(img).read()), dtype ="uint8")
			imgs = cv2.imdecode(imgs, cv2.IMREAD_GRAYSCALE)
			if pre_type == 'none':
				imgs = imgs
			elif pre_type == 'zero':
				imgs = zero_padding(imgs)
			elif pre_type == 'resize':
				imgs = np.array(cv2.resize(imgs, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)).astype('uint8')
				imgs = np.expand_dims(imgs, axis=-1)

			#Writing pre-processed train images to new folder 'pp_data_train'
			np.save(base_dir + "pp_data_train/" + file_name + '/img' + str(n) + ".npy", imgs)

def load_test(test_file, base_dir, dim_x=640, dim_y=640, n_frames=30):
	"""A funciton to unzip, preprocess, and store test image files as numpy arrays.
	
	:param test_file: a file containing test hashes
	:param base_dir: a directory where processed test images are saved
	:param dim_x: an integer identifying the width of the image
	:param dim_y: an integer identifying the length of the image
	:param n_frames: an integer specifying the number of frames per video
	:returns: None
	:raises NotADirectoryError: if /data cannot be located inside base_dir directory
	"""

	# checking if the test_file exist, if not download it from GC bucket
	if os.path.isfile(test_file):
		test_files = open(test_file).read().split('\n')
	else:
		test_files = requests.get('https://storage.googleapis.com/uga-dsp/project2/test.txt').text.split()

	# Checking if the data and masks directories are present in the project directory
	if not os.path.isdir(base_dir + "/data"):
		raise NotADirectoryError("data folder is missing in project directory, download the folder from uga-dsp bucket and run again")

	# Initializing list to store original test dimensions
	test_dimensions = []

	# Test files pre-processing and also populating list of original image sizes for each test hash
	for file_name in test_files:
		img_file = tarfile.open(base_dir + "/data/" + file_name + ".tar")
		extracted_img = img_file.getnames()[0:n_frames]
		
		# Creating directory to save test hashes
		if not os.path.isdir(base_dir + "pp_data_test"):
			os.mkdir(base_dir + "pp_data_test")
		
		# Creating directory for each individual training hash
		if not os.path.isdir(base_dir + "pp_data_test"):	
			os.mkdir(base_dir + "pp_data_test/" + file_name)
		
		for n, img in enumerate(extracted_img):
			imgs = np.asarray(bytearray(img_file.extractfile(img).read()), dtype ="uint8")
			imgs = cv2.imdecode(imgs, cv2.IMREAD_GRAYSCALE)
			
			# Appending original dimensions of image to list
			original_size = imgs.shape
			test_dimensions.append((file_name,original_size))
			
			if pre_type == 'none':
				imgs = imgs
			elif pre_type == 'zero':
				imgs = zero_padding(imgs)
			elif pre_type == 'resize':
				imgs = np.array(cv2.resize(imgs, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)).astype('uint8')
				imgs = np.expand_dims(imgs, axis=-1)
	 
			# Writing pre-processed test images to new folder 'pp_data_test'
			np.save(base_dir + "pp_data_test/" + file_name + '/img' + str(n) + ".npy", imgs)

	# Saving populated list of test hashes with original dimensions
	with open(base_dir + 'test_dimensions.pkl', 'wb') as f:
		pickle.dump(test_dimensions, f)
