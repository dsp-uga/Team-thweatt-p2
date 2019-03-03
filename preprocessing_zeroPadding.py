import cv2
import numpy as np
import os

dir = "/Users/vishakhaatole/Desktop/DSP/proj2data"

train_file = open(dir+"/"+"train.txt",)

train_names = []
for data in train_file:
    split_data = data.split()[0]
    train_names.append(split_data)

print(train_names)

def zero_padding(input_array, final_dimension = [640,640]):
    input_array_size = input_array.shape
    if input_array_size[0] != final_dimension[0] or input_array_size[1] != final_dimension[1]:
        temp = np.zeros((final_dimension[0],final_dimension[1]))
        temp[:input_array_size[0],:input_array_size[1]] = input_array
        return temp
    else:
        return input_array
