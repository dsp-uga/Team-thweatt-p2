import math
import os
import sys

import cv2
import scipy.misc
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from util import get_pp_images_test, get_pp_images_train
from util import get_features_test, get_features_train


def load_data(base_dir, n_frames=30):
    """ A function to load training and test data for classifier.

    :param base_dir: a directory where processed training images are saved.
    :param n_frames: an integer specifying the number of frames per video.
    :returns: array of training data, training labels, and test data.
    """

    train_dir = base_dir + "/pp_data_train"
    test_dir = base_dir + "/pp_data_test"
    masks_folder = base_dir + "/pp_masks"

    # Checking if the preprocessed train,test, and masks folders are
    # generated in the project directory
    if not os.path.isdir(train_dir):
        sys.exit("""Pre-Processed train folder is missing in project directory,
                 run the pre-process script to generate the folder""")
    if not os.path.isdir(masks_folder):
        sys.exit("""Pre-Processed mask folder is missing in project directory,
                 run the pre-process script to generate the folder""")
    if not os.path.isdir(test_dir):
        sys.exit("""Pre-Processed test folder is missing in project directory,
                 run the pre-process script to generate the folder""")

    # Reading train and masks
    X_train, y_train = get_pp_images_train(train_dir, masks_folder)

    # Reading test images and their hashes
    X_test, test_names = get_pp_images_test(test_dir)

    # Finding the number of training and testing hashes
    ntrain = len([f for f in os.listdir(train_dir) if not f.startswith('.')])
    ntest = len([f for f in os.listdir(test_dir) if not f.startswith('.')])

    # Converting X_train, y_train, and X_test to single numpy array and
    # reshaping
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (ntrain, n_frames))
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (ntest, n_frames))

    return X_train, y_train, X_test, test_names


def model_train(X_train, y_train, clf='rf'):
    """ A function to train a classifier.

    :param X_train: an array of training samples.
    :param y_train; an array of training labels.
    :param clf: a string specifying the classifier takes values 'rf' or 'svm's.
    :returns: a trained model.
    :raises ValueError: when the string is not 'rf' or 'svm'.
    """

    # Finding features for training set
    print("Extracting features from training images...")
    train_df = get_features_train(X_train, y_train)

    # Changing mask column to type category
    train_df['mask'] = train_df['mask'].astype('category')

    # Selecting train features
    feature_list = ['angle_of', 'mag_of', 'int_img', 'fft', 'var']
    x = train_df[feature_list]
    y = train_df['mask']

    # Training the classifier
    if clf == 'rf':
        mdl = RandomForestClassifier(n_estimators=100,
                                     max_depth=10,
                                     random_state=0,
                                     verbose=1,
                                     n_jobs=-1)
    elif clf == 'svm':
        mdl = SVC(C=1.0, cache_size=7000,
                  class_weight='balanced',
                  degree=3,
                  gamma=0.10000000000000001,
                  kernel='rbf',
                  probability=True)
    else:
        raise ValueError("clf paramter must be 'rf' or 'svm', %s given." % clf)

    print("training the classifier...")
    clf_fit = mdl.fit(x, y)
    return clf_fit


def model_predict(base_dir, X_test, test_names, trained_model):
    """ A function to make prediction on the test data

    :param base_dir: a directory where processed training images are saved.
    :param x_test: an array of test samples
    :param trained_model: a fitted model.
    :returns: None
    """

    # Creating output directory to write predictions
    if not os.path.isdir(base_dir + "/output"):
        os.mkdir(base_dir + "/output")

    feature_list = ['angle_of', 'mag_of', 'int_img', 'fft', 'var']
    # Predicting for each test image and writing to output directory
    for i in range(len(X_test)):
        print("Predicting on test sample " + str(i + 1))
        current_test = X_test[i]
        test_name = test_names[i]
        dim_x = current_test[1].shape[0]
        dim_y = current_test[1].shape[1]
        test_df = get_features_test(current_test)
        y_pred = trained_model.predict_proba(test_df[feature_list])
        y_pred_cut = np.where(y_pred[:, 1] > 0.40, 2, 0)
        pred_img = np.reshape(y_pred_cut, (dim_x, dim_y)).astype('uint8')
        scipy.misc.toimage(pred_img, cmin=0.0, cmax=...).\
            save(base_dir + '/output/' + test_name + '.png')
