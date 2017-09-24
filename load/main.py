from ear_pen import load, to_categorical_3d
import numpy as np

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = load()
    train_y = to_categorical_3d(train_y)