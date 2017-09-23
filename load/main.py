from ear_pen import load, to_categorical_3d
import numpy as np

(train_x, train_y), (test_x, test_y) = load()
train_y = to_categorical_3d(train_y)