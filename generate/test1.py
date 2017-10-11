import numpy as np

a = np.diag([1, 2, 3])
print(a)
print(a[:, ::-1])
print(a[::-1, :])
print(a[::-1, ::-1])