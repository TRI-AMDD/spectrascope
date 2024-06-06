# Code to extract features from the data

import numpy as np
import pandas as pd

from itertools import combinations


def degree_based_features(X,
                          degree,
                    exclude_features:list[str],
                          input_labels:list[str]):
    """
    Function to extract features from the data
    :param input:
    :param feature_types:
    :return:
    """

    #TODO implement degree >=3
    #TODO make dataframe implementation w/ labels
    new_features =[]
    if degree>=1:
        for x in X:

            new_features.append(x)
            new_features.append(np.log(x))
            new_features.append(np.sqrt(x))
            new_features.append(1/x)

    if degree>=2:

        for x in X:
            new_features.append(x*x)
            new_features.append(np.log(x)**2)
            new_features.append(np.sqrt(x)**3)
            new_features.append(1/x**2)
            new_features.append(np.log(x)/x)

        #Loop through pairs of values in X
        for xi,xj in combinations(X,2):
            new_features.append(xi*xj)

    return new_features


