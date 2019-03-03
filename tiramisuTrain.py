import cv2
import numpy as np
import os
import json


#Implementation based on https://github.com/0bserver07/One-Hundred-Layers-Tiramisu
dir = "/Users/vishakhaatole/Desktop/proj2data"

train_file = open(dir+"/"+"train.txt",)

i = 0
train_names = ""
for data in train_file:
    if i==0:
        split_data = data.split()[0]
        train_names = split_data
        i += 1

train_dir = dir + "/" + "Archive" + "/" + "pp_data_train_resize_512x512"
mask_dir = dir + "/" + "Archive" + "/" + "pp_masks_resize_512x512"

train_data = np.load(train_dir + "/" + train_names + "/" + "img0" +".npy")
#train_data = np.expand_dims(train_data, axis = 0)
train_data = train_data[None, ...]
#-train_data = np.squeeze(train_data,axis=3)
print(train_data.shape)


#-train_data = train_data[... , None]
#train_data = train_data.reshape(640, 640, 3)
#-train_label =  np.load(mask_dir + "/" + train_names +".npy")
#train_label = train_label[...,0]
#-train_label = train_label[None, ...]
#-train_label = np.squeeze(train_label,axis=3)
#-train_label = train_label[... , None]
#train_label = np.expand_dims(train_label, axis = 0)
#-print(train_data.shape)
#-print(train_label.shape)

'''train_dir = dir + "/" + "padding_intermediate"
mask_dir = dir + "/" + "padding_intermediate_masks"

train_data = np.load(train_dir + "/" + train_names + ".npy")
train_data = np.expand_dims(train_data, axis = 0)
train_data = train_data.reshape(640, 640, 3)
train_label =  np.load(mask_dir + "/" + train_names +".npy")
train_label = np.expand_dims(train_label, axis = 3)
print(train_data.shape)
print(train_label.shape)
'''

print(train_data.shape)
from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, Conv2DTranspose

from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.models import Model
from keras import regularizers

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K

from keras import callbacks
import math
K.set_image_dim_ordering('tf')
np.random.seed(7)
with open("/Users/vishakhaatole/Desktop" + "/" + "tiramisu_fc_dense103_model_1.json") as model_file:
    tiramisu = models.model_from_json(model_file.read())
from keras.callbacks import LearningRateScheduler
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.00001
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
lrate = LearningRateScheduler(step_decay)
optimizer = RMSprop(lr=0.001, decay=0.0000001)
tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
TensorBoard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=True)
filepath="weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2,save_best_only=True, save_weights_only=False, mode='max')


callbacks_list = [checkpoint]
nb_epoch = 150
batch_size = 2
history = tiramisu.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,callbacks=callbacks_list,verbose=1, shuffle=True) # validation_split=0.33
tiramisu.save_weights('weights/prop_tiramisu_weights_67_12_func_10-e7_decay{}.hdf5'.format(nb_epoch))
