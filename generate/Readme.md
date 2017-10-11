# Ear-Pen

[![Packagist](https://img.shields.io/badge/Python-3.5-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Numpy-1.13.1-blue.svg)]()

Usage
---
Use the following command to generate the original data file
```
# Generate the original file without compression
$ python build.py

# Generate the original file with compression scale of 2
$ python build.py --scale 2
```

Then, generate the extra data by imgaug. It's essential to install the imgaug before you run this generate script.    
```
# Install package
$ sudo pip install imgaug

# Generate extra image
$ python generate.py
```

At last, move the `.h5` file to the folder of your code. Regard it as the training data and start your training!    

Notice
---
The training data might be very large. In my experiment, after running the generate script with `--scale=2`, the size of `.h5` file is over 2 GB. You should notice if you have enough RAM to store the data.     