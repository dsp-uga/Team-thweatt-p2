import math
import os

import cv2
import numpy as np
import pandas as pd


def zero_padding(input_array, dim_x=640, dim_y=640):
    """ A functio to add trailing zeros to 2D numpy arrays.

    :param input_array: a 2D numpy array
    :param dim_x: an integer identifying the width of the image
    :param dim_y: an integer identifying the length of the image
    :returns: a 3D numpy array of shape (dim_x, dim_y, 1)
    """
    input_array_size = input_array.shape
    temp = np.zeros((dim_x, dim_y))
    temp[:input_array_size[0], :input_array_size[1]] = input_array
    temp = np.expand_dims(temp, axis=-1)
    return temp


def get_variance(imgs):
    """ A function to calculate the variance across a 3D numpy array.

    :param imgs: a 3D numpy array
    :returns: a 1D numpy array containing the variance values of imgs
    """
    imgs_shape = np.stack(imgs)
    var_imgs = np.var(imgs_shape, axis=0)
    var_flat = var_imgs.flatten()
    var_flat_scaled = np.interp(var_flat, (var_flat.min(),
                                var_flat.max()), (0, 1))
    return var_flat_scaled


def get_pp_images_train(train_folder, masks_folder):
    """ A function to prepare training data and masks. This function zero out
    the mask pixels that are not cilia.

    :param train_folder: a folder containing preprocessed training images.
    :param masks_folder: a folder containing preprocessed  mask images.
    :returns: array of training images and masks.
    """
    folders = [f for f in os.listdir(train_folder) if not f.startswith('.')]
    X_train, y_train = [], []
    for folder in folders:
        for file in os.listdir(train_folder + '/' + folder):
            X_train.append(np.load(train_folder + '/' + folder + '/' + file))
        mask = np.load(masks_folder + '/' + folder + '.npy')
        mask[np.where(mask != 2)] = 0
        y_train.append(mask)
    return X_train, y_train


def get_pp_images_test(test_folder):
    """ A function to prepare test data and labels.

    :param test_folder: a folder containing preprocessed test images.
    :returns: array of test images and labels.
    """
    folders = [f for f in os.listdir(test_folder) if not f.startswith('.')]
    X_test, y_test = [], []
    for folder in folders:
        for file in os.listdir(test_folder + '/' + folder):
            X_test.append(np.load(test_folder + '/' + folder + '/' + file))
        y_test.append(folder)
    return X_test, y_test


def get_features_train(X_train):
    """ A function to create a dataframe containing the following features:
        - angle_of: movement angle derived from optical flow.
        - mag_of: movement magnitude derived from optical flow.
        - int_img: actual intensity of each pixel.
        - fft: Fast Fourier Transform of each video.
        - var: variance of each pixel across all frames of a video.
        - mask: original mask.
    Equal rate sampling technique is used to create a balanced feature set.
    It takes 25% of background pixels and 75% of cilia pixels.

    :param X_train: entire array of training images.
    :returns: a dataframe containing all features.
    """

    # Creating empty dataframe to populate with train features
    train_dict = {'angle_of': [], 'mag_of': [], 'int_img': [],
                  'fft': [], 'var': [], 'mask': []}
    train_df = pd.DataFrame(train_dict)

    # Finding Optical flow for all frames related to the first frame and FFT
    for j in range(len(X_train)):
        current_train = X_train[j]
        current_mask = y_train[j]
        prvs = current_train[0]

        for i in range(1, len(current_train)):
            next = current_train[i]

            # Calculating optical flow
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5,
            	                                3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            angle = ang * 180 / np.pi / 2
            magn = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            # Calculating FFT
            dft = cv2.dft(np.float32(next), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0],
                                               dft_shift[:, :, 1])
            magnitude_spectrum_n = cv2.normalize(magnitude_spectrum, None,
                                                 0, 255, cv2.NORM_MINMAX)

            if i == 1:
                mags = magn
                angs = angle
                ffts = magnitude_spectrum_n
            else:
                mags = np.sum([mags, magn], axis=0)
                angs = np.mean([angs, angle], axis=0)
                ffts = np.mean([ffts, magnitude_spectrum_n], axis=0)

        # Flattening all array to form a single dataframe
        flat_angle = angs.flatten()
        flat_mag = mags.flatten()
        flat_int = prvs.flatten()
        flat_fft = ffts.flatten()
        flat_var = get_variance(current_train)
        flat_mask = current_mask.flatten()

        # Creating temporary dataframe for each hash
        dict_temp = {'angle_of': flat_angle, 'mag_of': flat_mag,
                     'int_img': flat_int, 'fft': flat_fft,
                     'var': flat_var, 'mask': flat_mask}
        df = pd.DataFrame(dict_temp)

        # Equal rate sampling
        df_2 = df[df['mask'] == 2]
        df_0 = df[df['mask'] == 0]
        if df_2.shape[0] != 0:
            df_cilia = df_2.sample(n=math.ceil(0.75 * df_2.shape[0]))
        else:
            df_cilia = df_2
        if df_0.shape[0] != 0:
            df_back = df_0.sample(n=math.ceil(0.25 * df_0.shape[0]))
        else:
            df_back = df_0
        train_df = pd.concat([train_df, df_cilia, df_back])
    return train_df


def get_features_test(X_test):
    """ A function to create a dataframe containing the following features:
        - angle_of: movement angle derived from optical flow.
        - mag_of: movement magnitude derived from optical flow.
        - int_img: actual intensity of each pixel.
        - fft: Fast Fourier Transform of each video.
        - var: variance of each pixel across all frames of a video.

    :param X_test: entire array of test images.
    :returns: a dataframe containing all features.
    """

    prvs = X_test[0]

    for i in range(1, len(X_test)):
        next = X_test[i]
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5,
                                            3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        angle = ang * 180 / np.pi / 2
        magn = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        dft = cv2.dft(np.float32(next), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0],
                                           dft_shift[:, :, 1])
        magnitude_spectrum_n = cv2.normalize(magnitude_spectrum, None,
                                             0, 255, cv2.NORM_MINMAX)

        if(i == 1):
            mags = magn
            angs = angle
            ffts = magnitude_spectrum_n
        else:
            mags = np.sum([mags, magn], axis=0)
            angs = np.mean([angs, angle], axis=0)
            ffts = np.mean([ffts, magnitude_spectrum_n], axis=0)

    flat_angle = angs.flatten()
    flat_mag = mags.flatten()
    flat_int = prvs.flatten()
    flat_fft = ffts.flatten()
    flat_var = get_variance(X_test)

    dict_temp = {'angle_of': flat_angle, 'mag_of': flat_mag,
                 'int_img': flat_int, 'fft': flat_fft, 'var': flat_var}

    test_df = pd.DataFrame(dict_temp)
    return test_df
