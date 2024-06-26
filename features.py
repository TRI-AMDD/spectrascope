# Code to extract features from the data

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
from itertools import combinations
from spectrascope.utils import find_nearest


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
        self.energies = np.array(energies)
        self.intensities = np.array(intensities)
        
    def pointwise_features(self, ne: int = 100, return_features: bool = False):
        """
        Interpolates self.energies, self.intensities onto ne evenly spaced points.
        
        Args:
            ne (int): The number of evenly spaced points to interpolate onto.
            return_features (bool): Flag indicating whether to return the features.
        """
        f = interp1d(self.energies, self.intensities)
        self.xinterp = np.linspace(min(self.energies), max(self.energies), ne)
        self.yinterp = f(self.xinterp)
        
        if return_features:
            return self.yinterp
    
    def generate_bins(self, N: int, interpolate: bool = True, x: np.array = None, y: np.array = None, verbose=False):
        """
        Generate N bins even in energy for self.energies.
        
        Args:
            N (int): The number of bins to generate.
            interpolate (bool): Flag indicating whether to interpolate the data before generating bins.
            x,y (np.array): Energy and intensity values to generate bins for. Default None uses self.energies and self.intensities.
        """
        if x is None: 
            # use default energies and intensities
            if interpolate: 
                if hasattr(self, 'xinterp') == False: 
                    self.pointwise_features()
                    
                x = self.xinterp
                y = self.yinterp
            else: 
                x = self.energies
                y = self.intensities
                
            use_default_energies = True
        else: 
            assert len(x) == len(y), 'x and y must be the same length'
            use_default_energies = False
            
            
        bin_width = (max(x) - min(x)) / N
        bin_indices = np.zeros(len(x), dtype=int)
        
        for i, val in enumerate(x):
            bin_indices[i] = int((val) - min(x)) / bin_width
            if bin_indices[i] == N:
                bin_indices[i] -= 1
                
        # if bin_indices.max() != N-1:
        #     raise ValueError('Cannot generate even bins with the given N on this range')
        if use_default_energies: 
            setattr(self, f'bin_indices_{N}', bin_indices)
            if verbose: 
                print(f'Generated {N} bins for energies across the full energy range, attribute: bin_indices_{N}')
        else: 
            setattr(self, f'bin_indices_{round(x.min())}_{round(x.max())}_{N}', bin_indices)
            if verbose: 
                print(f'Generated {N} bins for energies between {round(x.min(),2)} and {round(x.max(),2)}, attribute: bin_indices_{round(x.min())}_{round(x.max())}_{N}')
    def polynomial_featuresN(self, N: int, interpolate: bool = True, degree:int = 3, return_features: bool = False,verbose=True, **kwargs): 
        """
        Performs polynomial fitting on N energy equivalent sections.
        
        Args:
            N (int): The number of energy equivalent sections to perform polynomial fitting on.
            interpolate (bool): Flag indicating whether to interpolate the data before performing polynomial fitting.
            return_features (bool): Flag indicating whether to return the features.
        """
        featuresN = np.zeros([N,degree+1])

        use_default_energies = not 'x' in kwargs 
        if use_default_energies: 
            if interpolate: 
                if hasattr(self, 'xinterp') == False: 
                    self.pointwise_features()
                    
                x = self.xinterp
                y = self.yinterp
            else: 
                x = self.energies
                y = self.intensities
        else: 
            x = kwargs['x']
            y = kwargs['y']
        
        for i in range(0,N):
            # try: 
            self.generate_bins(N, **kwargs)
            # except ValueError: 
            #     print('Could not generate even bins with the given N on this energy range')
            #     continue 
            
            if use_default_energies: 
                bin_indices = getattr(self, f'bin_indices_{N}')
            else: 
                bin_indices = getattr(self, f'bin_indices_{round(x.min())}_{round(x.max())}_{N}')
            xbin = x[bin_indices==i]
            ybin = y[bin_indices==i] 
            binrange = (xbin.max() - xbin.min())
            
            #centering the x values
            xbin = xbin - xbin.mean()
            
            coefs = np.polyfit(xbin, ybin, degree)
            featuresN[i] = coefs
        
        if use_default_energies:
            setattr(self, f'polyfeatures_{N}_{degree}deg', featuresN)
            if verbose: 
                print(f'Generated {N} polynomial features using a degree {degree} polynomial for energies across the full energy range, attribute: polyfeatures_{N}_{degree}deg')
        else:
            setattr(self, f'polyfeatures_{round(x.min())}_{round(x.max())}_{N}_{degree}deg', featuresN)
            if verbose: 
                print(f'Generated {N} polynomial features using a degree {degree} polynomial for energies between {round(x.min(),2)} and {round(x.max(),2)}, attribute: polyfeatures_{round(x.min())}_{round(x.max())}_{N}_{degree}deg')
            
        if return_features:
            return featuresN
        
    def polynomial_featuresall(self, degree:int = 3, Narr: np.array = np.array([4,5,10,20]), interpolate: bool = True, return_features: bool = False, **kwargs): 
        """
        Performs polynomial fitting on multiple energy equivalent sections.
        
        Args:
            Narr (np.array): An array of integers representing the number of energy equivalent sections to perform polynomial fitting on.
            interpolate (bool): Flag indicating whether to interpolate the data before performing polynomial fitting.
            return_features (bool): Flag indicating whether to return the features.
            degree (int): The degree of the polynomial to fit.
        """
        self.polyfeatures_all = np.array([])
        
        for N in Narr: 
            
            self.polynomial_featuresN(N, interpolate, degree=degree, **kwargs)
            featuresN = getattr(self, f'polyfeatures_{N}_{degree}deg')
            self.polyfeatures_all = np.append(self.polyfeatures_all, featuresN.ravel())
            
            delattr(self, f'polyfeatures_{N}_{degree}deg')
            delattr(self, f'bin_indices_{N}')
        self.polyfeatures_all = np.append(self.polyfeatures_all, self.energies[np.argmax(self.intensities)])
        
        if return_features: 
            return self.polyfeatures_all
        

    def uneven_features(self, interpolate: bool = True, eV_splits: list[int] = [20,40], N_dict: dict = {0:10, 1:4, 2:5}, degree_dict: dict = {0:3,1:5,2:3}, verbose: bool =True): 
            """
            Split the data into uneven sections and perform polynomial fitting on each section 
            Option to split each section into different N values and degreess
            args: 
                    interpolate: bool, whether to interpolate the data before function 
                    eV_splits: list[int], the energy values to split the data at
                    N_dict: dict, the N values for each split {0:N, 1:N, 2:N}, must have len(ev_splits) + 1
                    degree_dict: dict, the degree values for each split {0:degree, 1:degree, 2:degree}, must have len(N_dict)
                    verbose: bool, whether to print names of attributes             
            """
            idx_arr = [find_nearest(self.energies, self.energies.min() + eV) for eV in eV_splits]
            assert len(N_dict) == len(eV_splits) + 1, 'There are not enough N values for the number of splits'
            assert len(degree_dict) == len(N_dict), 'There are not enough degree values for the number of N values'
                    
            if interpolate: 
                    if hasattr(self, 'xinterp') == False: 
                            self.pointwise_features()
                    x_full = self.xinterp
                    y_full = self.yinterp
            else: 
                    x_full = self.energies
                    y_full = self.intensities
                    
            for i,idx in enumerate(idx_arr): 
                    N = N_dict[i]
                    degree = degree_dict[i]
                    if i == 0: 
                            start, stop = 0, idx
                            x, y = x_full[start:stop], y_full[start:stop]
                            self.polynomial_featuresN(N, interpolate=False, degree=degree, x=x, y=y, verbose=verbose)
                    elif i < len(idx_arr):
                            start, stop = idx_arr[i-1], idx
                            x, y = x_full[start:stop], y_full[start:stop]
                            self.polynomial_featuresN(N, interpolate=False, degree=degree, x=x, y=y, verbose=verbose)
            N = N_dict[i+1]
            degree = degree_dict[i+1]
            start, stop = idx, -1  
            x, y = x_full[start:], y_full[start:]
            self.polynomial_featuresN(N, interpolate=False, degree=degree, x=x, y=y, verbose=verbose) 
            