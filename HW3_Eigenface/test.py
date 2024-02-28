import numpy as np

X = np.array([[1, 0.3, 0], [0.2, 1, 0.3], [0, 0.2, 1]])
q = np.array([0, 1, 0]).T
print(np.dot(np.linalg.inv(X), q))
