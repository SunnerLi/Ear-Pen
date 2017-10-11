from skimage import io, transform
import random
import numpy
import math

def flip(imgs, anns, prob=0.5):
    """
        Do the horizential flip and vertical filp with the specific probability

        Arg:    imgs    - The images array you want to flip
                anns    - The annotation array you want to flip
                prob    - The specific probability, default is 0.5
    """
    # Copy for original images and annotations
    res_imgs = []
    res_anns = []
    for i in range(len(imgs)):
        res_imgs.append(imgs[i])
        res_anns.append(anns[i])

    # Add flip images and annotation with specific probability
    for i in range(len(imgs)):
        if random.random() < prob:
            _flip_dir = random.randint(0, 3)
            if _flip_dir == 0:
                res_imgs.append(imgs[:, ::-1])
                res_anns.append(anns[:, ::-1])
            elif _flip_dir == 1:
                res_imgs.append(imgs[::-1, :])
                res_anns.append(anns[::-1, :])
            elif _flip_dir == 2:
                res_imgs.append(imgs[::-1, ::-1])
                res_anns.append(anns[::-1, ::-1])
    return np.asarray(imgs), np.asarray(anns)

def rotate(imgs, anns, prob=0.5):
    """
        Do the rotation with the specific probability
    """
    pass

def enlarge(imgs, anns, crop_ratio=0.7):
    """
        Crop the patch of 4 corner and central, and enlarge to the original size for each image
    """
    pass

def shrink(imgs, anns, shrink_min_ratio=0.5, prob=0.5):
    """
        Shrink the image to specific ratio, and padding with zero with the specific probability
    """
    pass