
import numpy as np

def zero_padding(input_array, final_dimension = [dimX,dimY]):
    input_array_size = input_array.shape
    temp = np.zeros((final_dimension[0],final_dimension[1]))
    temp[:input_array_size[0],:input_array_size[1]] = input_array
    temp = np.expand_dims(temp, axis=-1)
    return temp