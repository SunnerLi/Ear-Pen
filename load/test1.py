import numpy as np

a = np.ones([1, 10, 7, 3])
color = (1, 2, 3)
color = np.asarray(color)
a[-1, -1, -1, :] = color