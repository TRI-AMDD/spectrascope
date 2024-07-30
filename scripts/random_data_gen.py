import numpy as np
from numpy import ndarray

def generate_linear_5var_polynomial_data(X= None,noise=0.1)->tuple(ndarray,ndarray):
    # Where X is a vector of Nx5 variables
    if X is None:
        X = np.random.uniform(0, 10, (1000, 5))

    return X, (-2.8 * X[:, 0] - 2.7 * X[:, 1]
            - 5.3 * X[:, 2] + 4.3 * X[:, 3] + 9.0 * X[:, 4] + np.random.normal(loc=0, scale=noise, size=X.shape[0]))



X, Y = generate_linear_5var_polynomial_data(noise=0)




