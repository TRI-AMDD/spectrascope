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


class spectral_features:
    """
    A class that creates feature vectors extracted from XAS (X-ray Absorption Spectroscopy) data.

    Attributes:
        energies (np.array): Energy values from XAS spectrum.
        intensities (np.array): Intensity values from XAS spectrum.

    Methods:
        pointwise_features(ne: int = 100): Interpolates energies and intensities onto evenly spaced points.
        generate_bins(N: int, interpolate: bool = True): Generates N bins even in energy for polynomial featurization
        polynomial_featuresN(N: int, interpolate: bool = True): Performs polynomial fitting on N energy equivalent sections.
        polynomial_featuresall(Narr: np.array = np.array([4,5,10,20]), interpolate: bool = True): Performs polynomial fitting on a range of N energy equivalent sections.
    """
    def __init__(self, energies: np.array, intensities: np.array):
        """
        Initializes a new instance of the Spectral Features class.

        Args:
            energies (np.array): Energy values from XAS spectrum.
            intensities (np.array): Intensity values from XAS spectrum.
        """
        self.energies = energies
        self.intensities = intensities
        
    def pointwise_features(self, ne: int = 100):
        """
        Interpolates self.energies, self.intensities onto ne evenly spaced points.
        
        Args:
            ne (int): The number of evenly spaced points to interpolate onto.
        """
        f = interp1d(self.energies, self.intensities)
        self.xinterp = np.linspace(min(self.energies), max(self.energies), ne)
        self.yinterp = f(self.xinterp)
    
    def generate_bins(self, N: int, interpolate: bool = True):
        """
        Generate N bins even in energy for self.energies.
        
        Args:
            N (int): The number of bins to generate.
            interpolate (bool): Flag indicating whether to interpolate the data before generating bins.
        """
        if interpolate: 
            if hasattr(self, 'xinterp') == False: 
                self.pointwise_features()
                
            x = self.xinterp
            y = self.yinterp
        else: 
            x = self.energies
            y = self.intensities
            
        bin_width = (max(x) - min(x)) / N
        bin_indices = np.zeros(len(x), dtype=int)
        for i, val in enumerate(x):
            bin_indices[i] = int((val) - min(x)) / bin_width
            if bin_indices[i] == N:
                bin_indices[i] -= 1

        setattr(self, f'bin_indices_{N}', bin_indices)
        
    def polynomial_featuresN(self, N: int, interpolate: bool = True): 
        """
        Performs polynomial fitting on N energy equivalent sections.
        
        Args:
            N (int): The number of energy equivalent sections to perform polynomial fitting on.
            interpolate (bool): Flag indicating whether to interpolate the data before performing polynomial fitting.
        """
        featuresN = np.zeros([N,4])
        if interpolate: 
            if hasattr(self, 'xinterp') == False: 
                self.pointwise_features()
                
            x = self.xinterp
            y = self.yinterp
        else: 
            x = self.energies
            y = self.intensities
        
        for i in range(0,N):
            self.generate_bins(N)
            bin_indices = getattr(self, f'bin_indices_{N}')
            xbin = x[bin_indices==i]
            ybin = y[bin_indices==i] 
            binrange = (xbin.max() - xbin.min())
            
            #centering the x values
            xbin = xbin - xbin.mean()
            
            coefs = np.polyfit(xbin, ybin, 3)
            featuresN[i] = coefs
                

        setattr(self, f'polyfeatures_{N}', featuresN)
        
    def polynomial_featuresall(self, Narr: np.array = np.array([4,5,10,20]), interpolate: bool = True): 
        """
        Performs polynomial fitting on multiple energy equivalent sections.
        
        Args:
            Narr (np.array): An array of integers representing the number of energy equivalent sections to perform polynomial fitting on.
            interpolate (bool): Flag indicating whether to interpolate the data before performing polynomial fitting.
        """
        self.polyfeatures_all = np.array([])
        
        for N in Narr: 
            
            self.polynomial_featuresN(N, interpolate)
            featuresN = getattr(self, f'polyfeatures_{N}')
            self.polyfeatures_all = np.append(self.polyfeatures_all, featuresN.ravel())
            
            delattr(self, f'polyfeatures_{N}')
            delattr(self, f'bin_indices_{N}')
        self.polyfeatures_all = np.append(self.polyfeatures_all, self.energies[np.argmax(self.intensities)])
        