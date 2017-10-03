from utils import to_categorical_4d, to_categorical_4d_reverse
import ear_pen
import numpy as np

"""
    This main.py shows how to use the Ear-pen images in a simple way.
    Just like keras style, you can call `load_data` function to load data
    The returned data are two tuple with image and annotation

    Moreover, I provide `to_categorical_4d` function which is just like to_categorical in keras
    The different is that it will transfer the annotation as 4D tensor which channel number is the number of classes

    To be notice: the annotation should be normalized if you want to convert as one-hot format
"""

if __name__ == '__main__':
    # Load data and normalize
    (train_x, train_y), (test_x, test_y) = ear_pen.load_data()
    train_y = np.asarray(train_y)
    train_y = np.round(train_y/256, decimals=1)

    # to_categorical
    print('to_categorical_4d')
    train_y_, _map = to_categorical_4d(train_y)

    # to_categorical reverse
    print('to_categorical_4d_reverse')
    train_y_ = to_categorical_4d_reverse(train_y_, _map)
    
    # Check if there're the same
    print('origin shape: ', np.shape(train_y), '\tshape after reversed: ', np.shape(train_y_))
    print('Reversed result is the same? \t', np.array_equal(train_y, train_y_))