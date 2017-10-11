from skimage import io, transform
import numpy as np
import argparse
import h5py
import os

train_imgs = []
train_anns = []
test_imgs = []
test_anns = []

def nms(img, threshold=200):
    """
        Do the non-maximun supression toward the image

        Arg:    img         - The image you want to convert
                threshold   - The maximun threshold
    """
    res = np.zeros_like(img)
    idx = img > threshold
    res[idx] = 255
    return res

def add(path, list_obj, down_scale=1.0):
    """
        Add the image array to the specific list for the specific image path

        Arg:    path        - The path of the image
                list_obj    - The list object that you want to append
    """
    for img_name in os.listdir(path):
        img = io.imread(path + '/' + img_name)[:, :, :3]
        img = transform.rescale(img, 1.0 / float(down_scale), mode='constant')
        img = nms(img)
        list_obj.append(img)

if __name__ == '__main__':
    # Parse the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=1, dest='scale', help='shrink scale of the whole images')
    args = parser.parse_args()

    # Load images and annotations
    add('train/img', train_imgs, args.scale)
    add('train/tag', train_anns, args.scale)
    add('test/img', test_imgs, args.scale)
    add('test/tag', test_anns, args.scale)

    # Treat as numpy object
    train_imgs = np.asarray(train_imgs)
    train_anns = np.asarray(train_anns)
    test_imgs = np.asarray(test_imgs)
    test_anns = np.asarray(test_anns)

    # Save as .h5 file
    with h5py.File('ear_pen.h5', 'w') as f:
        f.create_dataset('train_x', data=train_imgs)
        f.create_dataset('train_y', data=train_anns)
        f.create_dataset('test_x', data=test_imgs)
        f.create_dataset('test_y', data=test_anns)
    print("the size of patches is: ", np.shape(train_imgs[0]))