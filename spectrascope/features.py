# Code to extract features from the data

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
from itertools import combinations
from spectrascope.utils import find_nearest, func_unfix_width_reg
import matplotlib.pyplot as plt
import pywt
from scipy.optimize import curve_fit

from scipy.fft import fft, fftfreq

from pyts.transformation import ShapeletTransform, BagOfPatterns, BOSS, ROCKET, WEASEL
from scipy.ndimage import gaussian_filter
import plotly.express as px

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA


def degree_based_features(
    X, degree, exclude_features: list[str], input_labels: list[str]
):
    """
    Function to extract features from the data
    :param input:
    :param feature_types:
    :return:
    """

    # TODO implement degree >=3
    # TODO make dataframe implementation w/ labels
    new_features = []
    if degree >= 1:
        for x in X:

            new_features.append(x)
            new_features.append(np.log(x))
            new_features.append(np.sqrt(x))
            new_features.append(1 / x)

    if degree >= 2:

        for x in X:
            new_features.append(x * x)
            new_features.append(np.log(x) ** 2)
            new_features.append(np.sqrt(x) ** 3)
            new_features.append(1 / x**2)
            new_features.append(np.log(x) / x)

        # Loop through pairs of values in X
        for xi, xj in combinations(X, 2):
            new_features.append(xi * xj)

    return new_features


class spectral_features:
    """
    A class that creates feature vectors extracted from XAS (X-ray Absorption Spectroscopy) data.

    Attributes:
        energies (np.array): Energy values from XAS spectrum.
        intensities (np.array): Intensity values from XAS spectrum.

    Methods:
        Every feature generation method assumes you want to perform on the interpolated spectrum
        pointwise_features(xrange): Interpolates energies and intensities onto evenly spaced points.
        generate_bins(N: int): Generates N bins even in energy for polynomial featurization
        polynomial_featuresN(N: int): Performs polynomial fitting on N energy equivalent sections.
        polynomial_featuresall(Narr: np.array = np.array([4,5,10,20])): Performs polynomial fitting on a range of N energy equivalent sections.
    """

    def __init__(self, energies: np.array, intensities: np.array):
        """
        Initializes a new instance of the Spectral Features class.

        Args:
            energies (np.array): x axis (energy, time)
            intensities (np.array): y axis (intensity, current, etc)
        """
        self.energies = np.array(energies)
        self.intensities = np.array(intensities)
        if len(self.intensities.shape) < 2:
            self.intensities = self.intensities.reshape(1, -1)

    def pointwise_features(
        self, xrange, return_features: bool = False, fill_value="extrapolate"
    ):
        """
        Interpolates self.energies, self.intensities onto ne evenly spaced points.

        Args:
            xrange (np.linspace(xmin,xmax)): range of x values to perform interpolation on
            return_features (bool): Flag indicating whether to return the features.
            fill_value (array, (tup), 'extrapolate') : fill value for scipy.interp1d for values outside of xrange (see scipy.interpolate.interp1d)
        """
        f = interp1d(
            self.energies,
            self.intensities,
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.xinterp = xrange
        self.yinterp = f(self.xinterp)
        self.de = np.diff(self.xinterp)[0]

        if return_features:
            return self.yinterp

    def generate_bins(
        self, N: int, x: np.array = None, y: np.array = None, verbose=False
    ):
        """
        Generate N bins even in energy for self.energies. If custom x and y not given, performs binning on
        interpolated data if it exists. Otherwise, performs on self.energies, self.intensities
        Args:
            N (int): The number of bins to generate.
            x,y (np.array, optional): Energy and intensity values to generate bins for. Default None uses full spectrum
        """
        if x is None:
            # use default energies and intensities
            # always perform functions on interpolated data
            x, y = self.check_xinterp(verbose)

            use_default_energies = True
        else:
            assert len(x) == y.shape[-1], "x and y must be the same length"
            use_default_energies = False

        bin_width = (max(x) - min(x)) / N
        bin_indices = np.zeros(len(x), dtype=int)
        # bin_indices = []

        for i, val in enumerate(x):
            index = int(((val) - min(x)) / bin_width)
            bin_indices[i] = index
            if bin_indices[i] == N:
                bin_indices[i] -= 1

        # if bin_indices.max() != N-1:
        #     raise ValueError('Cannot generate even bins with the given N on this range')
        if len(np.unique(bin_indices)) == len(np.arange(N)):
            if (np.unique(bin_indices) == np.arange(N)).all():

                if use_default_energies:
                    setattr(self, f"bin_indices_{N}", bin_indices)
                    if verbose:
                        print(
                            f"Generated {N} bins for energies across the full energy range, attribute: bin_indices_{N}"
                        )
                else:
                    setattr(
                        self,
                        f"bin_indices_{round(x.min())}_{round(x.max())}_{N}",
                        bin_indices,
                    )
                    if verbose:
                        print(
                            f"Generated {N} bins for energies between {round(x.min(),2)} and {round(x.max(),2)}, attribute: bin_indices_{round(x.min())}_{round(x.max())}_{N}"
                        )

        else:
            print("proper number of bins not generated, attribute not created")

    def polynomial_featuresN(
        self,
        N: int,
        degree: int = 3,
        return_features: bool = False,
        verbose=False,
        **kwargs,
    ):
        """
        Performs polynomial fitting on N energy equivalent sections.

        Args:
            N (int): The number of energy equivalent sections to perform polynomial fitting on.
            return_features (bool): Flag indicating whether to return the features.
        """
        featuresN = np.zeros([self.intensities.shape[0], 1])

        use_default_energies = not "x" in kwargs

        if use_default_energies:
            if hasattr(self, "xinterp") == False:
                if verbose:
                    print(
                        "generating features on self.energies, self.intensities...are these already interpolated? "
                    )
                x = self.energies
                y = self.intensities
            else:
                x = self.xinterp
                y = self.yinterp
        else:
            x = kwargs["x"]
            y = kwargs["y"]

        for i in range(0, N):

            if use_default_energies:
                try:
                    bin_indices = getattr(self, f"bin_indices_{N}")
                except AttributeError:
                    self.generate_bins(N, **kwargs)
                    bin_indices = getattr(self, f"bin_indices_{N}")
            else:
                try:
                    bin_indices = getattr(
                        self, f"bin_indices_{round(x.min())}_{round(x.max())}_{N}"
                    )
                except AttributeError:
                    self.generate_bins(N, **kwargs)
                    bin_indices = getattr(
                        self, f"bin_indices_{round(x.min())}_{round(x.max())}_{N}"
                    )

            xbin = x[bin_indices == i]
            ybin = y[:, bin_indices == i]
            binrange = xbin.max() - xbin.min()

            # centering the x values
            xbin = xbin - xbin.mean()

            coefs = np.polyfit(xbin, ybin.T, degree)
            featuresN = np.concatenate((featuresN, coefs.T), axis=1)

        featuresN = featuresN[:, 1:]

        if use_default_energies:
            setattr(self, f"polyfeatures_{N}_{degree}deg", featuresN)
            if verbose:
                print(
                    f"Generated {N*(degree+1)} polynomial features using a degree {degree} polynomial for energies across the full energy range split into {N} equivalent sections; attribute: polyfeatures_{N}_{degree}deg"
                )
        else:
            setattr(
                self,
                f"polyfeatures_{round(x.min())}_{round(x.max())}_{N}_{degree}deg",
                featuresN,
            )
            if verbose:
                print(
                    f"Generated {N*(degree+1)} polynomial features using a degree {degree} polynomial for energies between {round(x.min(),2)} and {round(x.max(),2)} split into {N} equivalent sections, attribute: polyfeatures_{round(x.min())}_{round(x.max())}_{N}_{degree}deg"
                )

        if return_features:
            return featuresN

    def polynomial_featuresall(
        self,
        degree: int = 3,
        Narr: np.array = np.array([4, 5, 10, 20]),
        interpolate: bool = True,
        return_features: bool = False,
        **kwargs,
    ):
        """
        Performs polynomial fitting on multiple energy equivalent sections.

        Args:
            Narr (np.array): An array of integers representing the number of energy equivalent sections to perform polynomial fitting on.
            interpolate (bool): Flag indicating whether to interpolate the data before performing polynomial fitting.
            return_features (bool): Flag indicating whether to return the features.
            degree (int): The degree of the polynomial to fit.
        """
        self.polyfeatures_all = np.zeros([self.intensities.shape[0], 1])

        for N in Narr:

            self.polynomial_featuresN(N, degree=degree, **kwargs)
            featuresN = getattr(self, f"polyfeatures_{N}_{degree}deg")
            self.polyfeatures_all = np.append(self.polyfeatures_all, featuresN, axis=1)

        self.polyfeatures_all = np.append(
            self.polyfeatures_all,
            self.energies[np.argmax(self.intensities, axis=1)].reshape(-1, 1),
            axis=1,
        )
        self.polyfeatures_all = self.polyfeatures_all[:, 1:]
        if return_features:
            return self.polyfeatures_all

    def uneven_features(
        self,
        eV_splits: list[int] = [20, 40],
        N_dict: dict = {0: 10, 1: 4, 2: 5},
        degree_dict: dict = {0: 3, 1: 5, 2: 3},
        verbose: bool = False,
        plot: bool = False,
        return_features: bool = False,
    ):
        """
        Split the data into uneven sections and perform polynomial fitting on each section
        Option to split each section into different N values and degrees
        args:
                interpolate: bool, whether to interpolate the data before function
                eV_splits: list[int], the energy values to split the data at
                N_dict: dict, the N values for each split {0:N, 1:N, 2:N}, must have len(ev_splits) + 1
                degree_dict: dict, the degree values for each split {0:degree, 1:degree, 2:degree}, must have len(N_dict)
                verbose: bool, whether to print names of attributes
                plot: plot subsections
        """
        idx_arr = [
            find_nearest(self.energies, self.energies.min() + eV) for eV in eV_splits
        ]

        assert (
            len(N_dict) == len(eV_splits) + 1
        ), "There are not enough N values for the number of splits"
        assert len(degree_dict) == len(
            N_dict
        ), "There are not enough degree values for the number of N values"

        x_full, y_full = self.check_xinterp(verbose)

        for i, idx in enumerate(idx_arr):
            N = N_dict[i]
            degree = degree_dict[i]
            if i == 0:
                start, stop = 0, idx
                x, y = x_full[start:stop], y_full[start:stop]
                self.polynomial_featuresN(N, degree=degree, x=x, y=y, verbose=verbose)
                if plot:
                    plt.plot(x, y)
            elif i < len(idx_arr):
                start, stop = idx_arr[i - 1], idx
                x, y = x_full[start:stop], y_full[start:stop]
                if plot:
                    plt.plot(x, y)
                self.polynomial_featuresN(N, degree=degree, x=x, y=y, verbose=verbose)
        N = N_dict[i + 1]
        degree = degree_dict[i + 1]
        start, stop = idx, -1
        x, y = x_full[start:], y_full[start:]
        self.polynomial_featuresN(N, degree=degree, x=x, y=y, verbose=verbose)
        if plot:
            plt.plot(x, y)
        if return_features:
            pass

    def cdf(self, verbose=False, return_features: bool = False):
        """
        Compute cumulative distribution function: probability that f(x) will take on a value less than or equal to x at x

        Args:
            intensity (arr): spectrum intensity
            verbose (bool) : print progress

        Attributes:
            cdf (np.array): cumulative distribution function for spectrum
        """
        x, y = self.check_xinterp(verbose)

        self.cdf_features = np.zeros_like(y)
        for idx, cumsum in enumerate(np.cumsum(y, axis=1)):
            self.cdf_features[idx] = np.array([i / np.max(cumsum) for i in cumsum])

        if return_features:
            return self.cdf_features

    def cwt(
        self,
        scales=np.arange(1, 11),
        wavelet="mexh",
        verbose=False,
        return_features: bool = False,
    ):
        """
        Compute continuous wavelet transform on spectrum: uses inner products to measure the similarity between a signal and an analyzing function (wavelet).
        Compare signal to shifted and compressed or stretched versions of a wavelet. Can choose different wavelets

        Args:
            intensity (arr): spectrum intensity
            scales (arr of ints, optional, default: np.arange(1,11)): _description_. Defaults to np.arange(1,11).
            wavelet (pywt.ContinuousWavelet, optional, default: 'mexh') : mother wavelet to compare spectrum to
            verbose (bool, default False) : print progress
        Attributes:
            coefs (2D np.array) : continuous wavlet transform of the imput signal for the given scales and wavelet. First axis corresponds to the scales,
            remaining axes match the shape of intensity
            frequencies: if the unit of sampling period are seconds and given, then frequencies are in herts, otherwise a sampling period of 1 assumed
        """
        x, y = self.check_xinterp(verbose)

        temp_coefs, self.cwt_frequencies = pywt.cwt(y, scales, wavelet, axis=1)
        cwt_shape = temp_coefs.shape

        # TODO : better way to reshape without for loop
        self.cwt_coefs = np.zeros([cwt_shape[1], cwt_shape[0], cwt_shape[-1]])
        for i in np.arange(cwt_shape[1]):
            self.cwt_coefs[i] = temp_coefs[:, i]

        if return_features:
            return self.cwt_coefs.reshape(cwt_shape[1], cwt_shape[0] * cwt_shape[2])

    def gaussian_peaks(self, num_peaks=20, verbose=False, return_features=True):
        """
        Fit {num_peaks} gaussians to intensity

        Args:
            num_peaks (int, default 20): Number of gaussians to fit
            verbose (bool, default False): print progress
        """
        self.peak_features = np.array([])
        x, y = self.check_xinterp(verbose)
        # TODO: vectorize

        for yi in y:
            idxes = [int(i) for i in np.linspace(0, len(x) - 1, num_peaks + 1)]
            idxes_2 = [
                int((idxes[i] + idxes[i + 1]) / 2) for i in range(len(idxes) - 1)
            ]
            guess = np.ravel([[x[i], yi[i], 2] for i in idxes_2])
            lower = np.ravel([[x[i], min(yi), 0] for i in idxes[:-1]])
            upper = np.ravel([[x[i], max(yi), 6] for i in idxes[1:]])
            popt, pcov = curve_fit(
                func_unfix_width_reg, x, yi, p0=guess, bounds=(lower, upper)
            )

            self.peak_features = np.append(self.peak_features, popt)

        if return_features:
            return popt

    def fft_features(self, return_features: bool = False, **kwargs):
        """
        FFT Spectra and return positive real FT values
        Args:
        """
        if hasattr(self, "xinterp") == False:
            if "xrange" not in kwargs:
                print(
                    "to perform fft need evenly spaced data, please provide xrange to perform interpolation need to perform interpolation"
                )
                return
            else:
                self.pointwise_features(kwargs["xrange"])

        N = len(self.xinterp)
        self.fft_feat = fft(self.yinterp)[:, : N // 2]
        self.fft_freq = np.abs(fftfreq(N, self.de))[: N // 2]

        if return_features:
            return self.fft_feat

    def gaussian_filter_features(
        self, sigma, verbose=False, return_features=False, **kwargs
    ):

        x, y = self.check_xinterp(verbose)

        self.gaussian_filter = gaussian_filter(y, sigma, **kwargs)

        if return_features:
            return self.gaussian_filter

    def gaussian_mixture_features(
        self, n_components, return_features: bool = False, **kwargs
    ):
        pass

    def pca(self, n_components, verbose=False, return_features: bool = False, **kwargs):
        x, y = self.check_xinterp(verbose)
        pca = PCA(n_components=n_components, **kwargs)
        self.pca_features = pca.fit_transform(y)

        if return_features:
            return self.pca_features

    def featureagg(
        self, n_clusters, verbose=False, return_features: bool = False, **kwargs
    ):
        x, y = self.check_xinterp(verbose)

        fa = FeatureAgglomeration(n_clusters=n_clusters, **kwargs)
        self.fa_features = fa.fit_transform(y)

        if return_features:
            return self.fa_features

    def b_splines(self):
        pass

    def shapelet(self, labels=None, verbose=False, return_features=False, **kwargs):
        """
        Transforms a data set to identify shapelets. Only works for classification?

        """
        x, y = self.check_xinterp(verbose)
        st = ShapeletTransform()
        self.shapelet_features = st.fit_transform(X=y, y=labels, **kwargs)
        if return_features:
            return self.shapelet_features

    def bow_features(self, **kwargs):
        bow = BagOfWords(**kwargs)
        pass

    def bop_features(self, **kwargs):
        bop = BagOfPatterns(**kwargs)
        pass

    def boss_features(self, **kwargs):
        boss = BOSS(**kwargs)
        pass

    def WEASEL_features(self, **kwargs):
        wsl = WEASEL(**kwargs)
        pass

    def ROCKET_featrues(self, **kwargs):
        rkt = ROCKET(**kwargs)
        pass

    ### Plotting functions for app
    ### May fail with current new updates
    def plot_gaussian_peaks(self, ax=None, verbose=False, **kwargs):

        x, y = self.check_xinterp(verbose)

        assert hasattr(
            self, "peak_features"
        ), "need to create gaussian peak features first"

        ax.plot(
            x,
            func_unfix_width_reg(x, *self.peak_features),
            label="Reconstructed spectrum",
        )
        for i in range(0, len(self.peak_features), 3):
            ctr = self.peak_features[i]
            amp = self.peak_features[i + 1]
            wid = self.peak_features[i + 2]
            yfit = amp * np.exp(-((x - ctr) ** 2 / (2 * (wid / 2.355) ** 2)))
            ax.plot(x, yfit, linestyle="-.", color="grey", **kwargs)

        fitted = np.array(fitted).ravel()
        return np.arange(len(fitted)), fitted

    def plot_cwt(self, ax=None, verbose=False, **kwargs):
        x, _ = self.check_xinterp(verbose)

        assert hasattr(self, "cwt_frequencies"), "need to create cwt features first"
        if ax is not None:
            ax.pcolormesh(x, self.cwt_frequencies, self.cwt_coefs, **kwargs)
        else:
            return self.cwt_frequencies, self.cwt_coefs

    def plot_cdf(self, ax=None, verbose=False, **kwargs):
        x, _ = self.check_xinterp(verbose)

        assert hasattr(self, "cdf_feat"), "need to create cdf features first"

        if ax is not None:
            ax.plot(x, self.cdf_feat)
        else:
            return x, self.cdf_feat

    def plot_polynomial_featuresall(self, ax=None, verbose=False, **kwargs):
        _, _ = self.check_xinterp(verbose)

        assert hasattr(self, "polyfeatures_all"), "need to create poly features first"

        x = np.arange(len(self.polyfeatures_all))
        if ax is not None:
            ax.plot(x[:-1], self.polyfeatures_all[:-1])
            ax.set_xlabel("Feature Index")

        else:
            return x[:-1], self.polyfeatures_all[:-1]

    def check_xinterp(self, verbose=False):
        if hasattr(self, "xinterp") == False:
            if verbose:
                print(
                    "generating features on self.energies, self.intensities...are these already interpolated? "
                )
            x = self.energies
            y = self.intensities
        else:
            x = self.xinterp
            y = self.yinterp
        return x, y
