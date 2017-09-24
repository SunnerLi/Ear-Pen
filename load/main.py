from ear_pen import load, to_categorical_3d, to_categorical_3d_reverse
import numpy as np

if __name__ == '__main__':
    # Load data and normalize
    (train_x, train_y), (test_x, test_y) = load()
    train_y = np.asarray(train_y)
    train_y = np.round(train_y/256, decimals=1)

    # to_categorical
    print('to_categorical_3d')
    train_y_, _map = to_categorical_3d(train_y)

    
    # to_categorical reverse
    print('to_categorical_3d_reverse')
    train_y_ = to_categorical_3d_reverse(train_y_, _map)

    # Check if there're the same
    print(np.shape(train_y), np.shape(train_y_))
    
    batch, height, width, channel = np.shape(train_y)
    for i in range(batch):
        for j in range(height):
            for k in range(width):
                for l in range(channel):
                    if train_y[i][j][k][l] != train_y_[i][j][k][l]:
                        print('position: ', i, j, k, l, '\torigin: ', train_y[i][j][k][l], '\tafter: ', train_y_[i][j][k][l])    
                        exit()