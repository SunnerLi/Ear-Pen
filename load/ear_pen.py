import numpy as np
import h5py
import os

pallete = dict()            # Mapping from color -> index
reverse_pallete = dict()    # Mapping from index -> color

def load():
    if not os.path.exists('./ear_pen.h5'):
        print('You should generate the ear_pen hdf5 file first...')
        exit()
    f = h5py.File('./ear_pen.h5')
    return (f['train_x'], f['train_y']), (f['test_x'], f['test_y'])

def to_categorical_3d(_input_tensor, pallete=None):
    """
        Change the representation to the one-hot format
        Just like the keras' to_categorical but in 3D shae

        Arg:    _input_tensor   - The input tensor want to change into one-hot format
        Ret:    The result tensor in 4D shape [batch_num, height, width, class_num]
    """ 
    _result_tensor = np.copy(_input_tensor)
    __buildPallete(_result_tensor, pallete)
    for i in range(np.shape(_result_tensor)[0]):
        for j in range(np.shape(_result_tensor)[1]):
            for k in range(np.shape(_result_tensor)[2]):
                _result_tensor[i][j][k] = pallete[_input_tensor[i][j][k]]
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
        pallete = dict()
    if len(np.shape(_input_tensor)) != 4:
        print('< ear_pen >  error: The shape of tensor should be 4!')
        exit()
    global reverse_pallete
    for _slice in _input_tensor:
        for i in range(np.shape(_slice)[0]):
            for j in range(np.shape(_slice)[1]):
                print()                                 # bug, the color is 3, and it cannot become index of dict
                if not _slice[i][j] in pallete:
                    pallete[_slice[i][j]] = len(pallete)
    reverse_pallete = {pallete[x]: x for x in pallete}