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



dir = "/Users/hemanth/Desktop/MSAI/DataSciencePracticum/Projects/p2/"
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
if os.path.exists(test_folder):
    X_test = get_pp_images(test_folder,masks_folder,mask = False)
else:
    print('Pre-processed testing set not available')
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

img = np.squeeze(y_train[90])

print(np.amin(img))
print(np.unique(img))
print(np.amax(img))
#print(stats.describe(img))
#img[np.where(img > 0.5)] = 2 

print(img.shape)
np.set_printoptions(threshold=np.nan)




assert X_train.shape == y_train.shape
assert len(X_train) == len(y_train)

#cv2.imshow('img',X_train[1])
#cv2.waitKey(0)
#cv2.imshow('img',y_train[1])
#cv2.destroyAllWindows()
    
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

inputs = Input((dimX, dimY, 1))
s = Lambda(lambda x: x / dimY-1)(inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
# model.summary()

#from keras.utils import plot_model

#plot_model(model, to_file='images/model_unet.png', show_shapes=True)

earlystopper = EarlyStopping(patience=5, verbose=1)
# checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=8, callbacks=[earlystopper])

