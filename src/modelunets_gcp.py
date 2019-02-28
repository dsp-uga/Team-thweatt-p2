import numpy as np
import warnings
import cv2
import os

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers import *
from keras.optimizers import *

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


dir = "/home/hemanthreg/dip-p2/"
train_folder = dir+"pp_data_train_resize_512x512"
test_folder = dir+"pp_data_test_resize_512x512"
masks_folder = dir+"pp_masks_resize_512x512"
dimX = 512
dimY = 512


def get_pp_images(train_folder,masks_folder,mask):
    folders = [f for f in os.listdir(train_folder) if not f.startswith('.')]
    X_train = []
    y_train = []
    for folder in folders:
        for file in os.listdir(train_folder+'/'+folder):
            X_train.append(np.load(train_folder+'/'+folder+'/'+file))
            if(mask == True):
                y_train.append(np.load(masks_folder+'/'+folder+'.npy'))
            else:
                y_train.append(folder)            
    return X_train,y_train

if os.path.exists(train_folder) and os.path.exists(masks_folder):
    X_train,y_train = get_pp_images(train_folder,masks_folder,mask = True)
else:
    print('Pre-processed training/masks set not available')
    
#Reading test images
if os.path.exists(test_folder):
    X_test,test_labels = get_pp_images(test_folder,masks_folder,mask = False)
else:
    print('Pre-processed testing set not available')

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)


y_train_copy = y_train

y_train[np.where(y_train < 2)] = 0 


assert X_train.shape == y_train.shape
assert len(X_train) == len(y_train)

def mean_iou(y_true, y_pred):
    prec = []
    y_true = y_true/255
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def unet(pretrained_weights = None,input_size = (dimX,dimY,1)):
    with tf.device('/device:GPU:0'):
        inputs = Input(input_size)
        s = Lambda(lambda x: x /255)(inputs)
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s)
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.3)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.3)(conv5)

        up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
        #model.summary()

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

    return model


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
model = unet()

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#earlystopper = EarlyStopping(patience=5, verbose=1)

model.fit(X_train,y_train,validation_split=0.2,batch_size=4,epochs=10,callbacks=[model_checkpoint])

# Predict on test
model = load_model('unet_membrane.hdf5', custom_objects={'mean_iou': mean_iou})
preds_test = model.predict(X_test, verbose=1)

import pickle

with open(dir+'test_dimensions.pkl', 'rb') as f:
    test_dimensions = pickle.load(f)

from skimage.transform import resize

# Threshold predictions
#preds_test_t = np.where(preds_test > 0.5,2,0).astype(np.uint8)
#preds_test[np.where(preds_test > 0.5)] = 2 

import imageio

if not os.path.isdir(dir+"output"):
    os.mkdir(dir+"output")

np.set_printoptions(threshold=np.nan)

if not os.path.isdir(dir+"output"):
    os.mkdir(dir+"output")

from skimage.transform import resize
    
for n,file in enumerate(test_labels):
    img = np.squeeze(preds_test[n])
    img[np.where(img > 0.5)] = 2 
    test_rec = [item for item in test_dimensions if item[0] == file]
    dim_X = test_rec[0][1][0]
    dim_Y = test_rec[0][1][1]
    img1 = np.array(cv2.resize(img, (int(dim_X), int(dim_Y)),interpolation=cv2.INTER_NEAREST)).astype('uint8')
    #img = img[:dimX,:dimY].astype(np.uint8)
    cv2.imwrite(dir+'/output/'+file+'.png',img1)
    
