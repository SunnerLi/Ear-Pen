from collections import Counter, OrderedDict
import numpy as np
import h5py
import os
import re

# Encode-Decode transformation function
ENCODE = lambda color: color[0] * 100 + color[1] * 10 + color[2]
DECODE = lambda color: [round((color-color%10)/100), round((color-color%1)%100/10), color%1]

pallete = dict()            # Mapping from color -> index
reverse_pallete = dict()    # Mapping from index -> color

def load():
    if not os.path.exists('./ear_pen.h5'):
        print('You should generate the ear_pen hdf5 file first...')
        exit()
    f = h5py.File('./ear_pen.h5')
    return (f['train_x'], f['train_y']), (f['test_x'], f['test_y'])

def normalize(_input_tensor):
    return np.round(_input_tensor/256, decimals=1)

def to_categorical_3d(_input_tensor, pallete=None):
    """
        Change the representation to the one-hot format
        Just like the keras' to_categorical but in 3D shae

        Arg:    _input_tensor   - The input tensor want to change into one-hot format
        Ret:    The result tensor in 4D shape [batch_num, height, width, class_num]
    """ 
    _result_tensor = np.zeros_like(_input_tensor)
    #_result_tensor = normalize(_input_tensor)
    pallete = __buildPallete(_input_tensor, pallete)

    # Build index tensor
    encode_map = _input_tensor[:, :, :, 0] * 100 + _input_tensor[:, :, :, 1] * 10 + _input_tensor[:, :, :, 2]
    encode_map = np.vectorize(pallete.get)(encode_map)

    # Fill the corresponding index as 1
    for _c in range(len(pallete)):
        _result_tensor[encode_map == _c] = 1
    return _result_tensor, pallete

def to_categorical_3d_reverse(_input_tensor, pallete):
    reverse_pallete = {pallete[x]: x for x in pallete}
    decode_map = np.argmax(_input_tensor, axis=-1)
    decode_map = np.vectorize(reverse_pallete.get)(decode_map)

    _result_tensor = np.zeros_like(_input_tensor)
    print('reverse: ', reverse_pallete)
    _result_tensor[:, :, :, 2] = decode_map[:, :, :] % 2
    decode_map[:, :, :] = (decode_map[:, :, :] - _result_tensor[:, :, :, 2]) / 10
    _result_tensor[:, :, :, 1] = decode_map[:, :, :] % 2
    decode_map[:, :, :] = (decode_map[:, :, :] - _result_tensor[:, :, :, 1]) / 10
    _result_tensor[:, :, :, 0] = decode_map[:, :, :] % 2
    
    return _result_tensor

def __buildPallete(_input_tensor, pallete):
    """
        Build the mapping of color and categorical index
        The warning will raise if you had built the mapping
        This function only accepts 4D tensor
    """
    if pallete != None:
        print('< ear_pen >  warning: You had built the pallete mapping')
    else:
        pallete = OrderedDict()
    if len(np.shape(_input_tensor)) != 4:
        print('< ear_pen >  error: The shape of tensor should be 4!')
        exit()
    for _slice in _input_tensor:
        encode_map = _slice[:, :, 0] * 100 + _slice[:, :, 1] * 10 + _slice[:, :, 2]
        counter = Counter(np.reshape(encode_map, [-1]))
        for key in counter:
            if not key in pallete:
                pallete[key] = len(pallete)
    return pallete