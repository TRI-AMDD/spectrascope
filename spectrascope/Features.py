"""
Spectral Features class for Spectrascope

"""

import numpy as np
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
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.gaussian_process import GaussianProcessRegressor

__docformat__ = "numpy"


class SpectralFeatures:
    """Spectral Features class used by SpectraScope main class to generate features on datasets.

    A class that creates feature vectors extracted from 1D array data.
    Current features developed for spectral data, but could be applied to any 1D data.
    Every feature generation method assumes you want to perform on the interpolated spectra, so
    looks for interpolated data before performing operations on the original data.
    Most featurizer functions are vectorized and can accept multiple samples at once.

    """

    def __init__(self, energies, intensities):
        """
        Parameters
        ----------
            energies : array-like
                x axis (energy, time). shape: (n_features,))
            intensities : array-like
                y axis (intensity, current, etc). shape: (n_samples, n_features))
        """

        self.energies = np.array(energies)
        self.intensities = np.array(intensities)
        if len(self.intensities.shape) < 2:
            self.intensities = self.intensities.reshape(1, -1)

    def pointwise_features(
        self, xrange, return_features: bool = False, fill_value="extrapolate"
    ):
        """Interpolates self.energies, self.intensities onto ne evenly spaced points.

        Parameters
        ----------
            xrange : np.array
                range of x values to perform interpolation on. (np.linspace(xmin,xmax))
            return_features : bool, default=False
                Flag indicating whether to return the features
            fill_value:  array-like or (array-like, array_like) or “extrapolate”, optional, default="extrapolate"
                fill value for scipy.interp1d for values outside of xrange (see scipy.interpolate.interp1d).
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
        """Generate N bins for polynomial fitting.

        Generate N bins even in energy for self.energies. If custom x and y not given, performs binning on
        interpolated data if it exists. Otherwise, performs on self.energies, self.intensities.

        Parameters
        ----------
        N : int
            The number of bins to generate.
        x : np.array, optional
            Energy values to generate bins for. Default None uses full spectrum.
        y : np.array, optional
            Intensity values to generate bins for. Default None uses full spectrum.
        verbose : bool, optional
            print progress. Default False.

        Attributes
        ----------
            bin_indices_{N} : np.array
                An array of bin indices for the N bins. Shape is (len(x),).
            bin_indices_{round(x.min())}_{round(x.max())}_{N} : np.array
                An array of bin indices for N bins over a subset of full energy range. Shape is (len(x),).
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
        """Performs polynomial fitting on N energy equivalent sections.

        Parameters
        ----------
        N : int
            The number of energy equivalent sections to perform polynomial fitting on.
        degree : int, optional, default=3
            The degree of the polynomial to fit.
        return_features : bool, optional, default=False
            Flag indicating whether to return the features.
        verbose : bool, optional, default=False
            print progress.

        Attributes
        ----------
        polyfeatures_{N}_{degree}deg : np.array
            An array of polynomial coefficients for the N bins. Shape is (N, degree + 1).
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
        """Performs polynomial fitting on multiple energy equivalent sections.

        Parameters
        ----------
        degree : int, optional, default=3
            The degree of the polynomial to fit.
        Narr : np.array, optional, default=[4, 5, 10, 20]
            An array of integers representing the number of energy equivalent sections to perform polynomial fitting on.
        interpolate : bool, optional, default=True
            Flag indicating whether to interpolate the data before performing polynomial fitting.
        return_features : bool, optional, default=False
            Flag indicating whether to return the features.
        **kwargs: dict
            additional parameters to pass to polynomial_featuresN.

        Attributes
        ----------
        polyfeatures_all : np.array
            An array of polynomial coefficients for the bins. Shape is sum((degree + 1) * n for n in Narr) + 1 (max energy position).
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
        """Split the data into uneven sections and perform polynomial fitting on each section.

        Option to split each section into different N values and degrees. Generates polynomial_featuresN(N, degree=degree, x=x, y=y, verbose=verbose) for every N, degree on each eV range (np.array).

        Parameters
        ----------
        eV_splits :  list, optional, default=[20, 40]
            list of ints, the energy values to split the data at.
        N_dict : dict, optional, default={0:10, 1:4, 2:5}
            the N values for each split {0:N, 1:N, 2:N}, must have len(ev_splits) + 1.
        degree_dict : dict, optional, default={0:3, 1:5, 2:3}
            the degree values for each split {0:degree, 1:degree, 2:degree}, must have len(N_dict).
        verbose : bool, optional, default=False
            whether to print names of attributes.
        plot : bool, optional, default=False
            plot subsections.
        return_features : bool, optional, default=False
            Flag indicating whether to return the features. Currently not implemented.
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
        """Compute cumulative distribution function: probability that f(x) will take on a value less than or equal to x at x.

        Parameters
        ----------
        verbose : bool, optional, default=False
            print progress.
        return_features : bool, optional, default=False
            return features.

        Attributes
        ----------
        cdf_features : np.array
            cumulative distribution function for spectrum.
        """

        x, y = self.check_xinterp(verbose)

        self.cdf_features = np.zeros_like(y)
        for idx, cumsum in enumerate(np.cumsum(y, axis=1)):
            self.cdf_features[idx] = np.array([i / np.max(cumsum) for i in cumsum])

        if return_features:
            return self.cdf_features

    def tsne(
        self, n_components=2, verbose=False, return_features: bool = False, **kwargs
    ):
        """T-distributed Stochastic neighbor embedding

        Converts similarities between data points to joint probabilities and tries to minimize
        the Kullback-Leibler divergence between the joint probabilties of the low-dimensional embedding
        and the high-dimensional data. t-SNE has a cost function that is not convex,
        i.e. with different initializations we can get different results.
        Recommended to apply PCA first to reduce dimensions (to ~50) then peform TSNE on the PCA features.

        Parameters
        ----------
        n_components : int, optional, default=2
            Number of components. Must be < 4.
        verbose : bool, optional, default=False
            print progress.
        return_features : bool, optional, default=False
            return features.
        **kwargs: dict
            additional parameters to pass to TSNE.

        Attributes
        ----------
        tsne_features : np.array
            tsne features shape (n_samples, n_components).
        """

        x, y = self.check_xinterp(verbose)
        tsne = TSNE(n_components=n_components, **kwargs)
        self.tsne_features = tsne.fit_transform(y)

        if return_features:
            return self.tsne_features

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

        Parameters
        ----------
        scales : np.array, optional, Default= np.arange(1,11).
            Array of ints.
        wavelet : pywt.ContinuousWavelet, optional
            Mother wavelet to compare spectrum to. Default 'mexh'.
        verbose : bool, optional
            print progress. Default False.
        return_features : bool, optional, Default=False.
            return features. Returns coefs raveled to len(widths) * len(intensity) to give 1D array for each sample returns shape (n_samples, len(widths)*len(intensity)) for SpectraScope class.

        Attributes
        ----------
        cwt_coefs : np.array
            continuous wavelet transform of the imput signal for the given scales and wavelet. First axis corresponds to the scales, remaining axes match the shape of intensity.
        cwt_frequencies : np.array
            if the unit of sampling period are seconds and given, then frequencies are in herts, otherwise a sampling period of 1 assumed.
        """

        x, y = self.check_xinterp(verbose)

        scales = np.array(scales).astype(int)

        temp_coefs, self.cwt_frequencies = pywt.cwt(y, scales, wavelet, axis=1)
        cwt_shape = temp_coefs.shape

        # reshape temp_coefs from (scales, samples, time) to (samples, scales, time)
        self.cwt_coefs = np.zeros([cwt_shape[1], cwt_shape[0], cwt_shape[-1]])
        for i in np.arange(cwt_shape[1]):
            self.cwt_coefs[i] = temp_coefs[:, i]

        if return_features:
            return self.cwt_coefs.reshape(cwt_shape[1], cwt_shape[0] * cwt_shape[2])

    def gaussian_peaks(self, num_peaks=20, verbose=False, return_features=True):
        """
        Fit n=num_peaks gaussians to intensity.
        Not vectorized yet.

        Parameters
        ----------
        num_peaks : int, optional, Default = 20
            Number of gaussians to fit.
        verbose : bool, optional
            print progress. Default False.
        return_features : bool, optional
            return features. Default True.

        Attributes
        ----------
        peak_features : np.array
            peak features. length = num_peaks * 3.
        """

        self.peak_features = np.array([])
        x, y = self.check_xinterp(verbose)

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
            return self.peak_features

    def fft_features(self, return_features: bool = False, **kwargs):
        """
        FFT Spectra and return positive real FFT values.

        Parameters
        ----------
        reutrn_features : bool, optional
            return features. Default False.
        **kwargs: dict
            additional parameters to pass to fft.

        Attributes
        ----------
        fft_feat : np.array
            fft features. shape (n_samples, n_features/2).
        fft_freq : np.array
            fft frequencies. shape (n_features/2,).
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
            return np.abs(self.fft_feat)

    def gaussian_filter_features(
        self, sigma, verbose=False, return_features=False, **kwargs
    ):
        """Convolve intensity with a gaussian filter.

        Parameters
        ----------
        sigma : float
            standard deviation of the gaussian filter.
        verbose : bool, optional
            print progress. Default False.
        return_features : bool, optional
            return features. Default False.
        **kwargs: dict
            additional parameters to pass to gaussian

        Attributes
        ----------
        gausian_filter : np.array
            gaussian filter features.
        """

        x, y = self.check_xinterp(verbose)

        self.gaussian_filter = gaussian_filter(y, sigma, **kwargs)

        if return_features:
            return self.gaussian_filter

    def gaussian_mixture_features(
        self, n_components: int, return_features: bool = False, **kwargs
    ):
        """Representation of a Gaussian mixture model probability distribution.

        Not impplemented yet.

        Parameters
        ----------
        n_components : int
            number of mixture components.
        return_features : bool, optional
            return features. Default False.
        **kwargs: dict
            additional parameters to pass to Gaussian

        Attributes
        ----------

        """

        pass

    def pca(
        self,
        n_components: int = 10,
        verbose=False,
        return_features: bool = False,
        **kwargs,
    ):
        """Principle component analysis

        Parameters
        ----------
        n_components : int, optional
            Number of PCA components. Default 10.
        verbose : bool, optional
            print progress. Default False.
        return_features : bool, optional
            return features. Default False.
        **kwargs: dict
            additional parameters to pass to PCA.

        Attributes
        ----------
        pca_features : np.array
            pca features.
        """

        x, y = self.check_xinterp(verbose)
        pca = PCA(n_components=n_components, **kwargs)
        self.pca_features = pca.fit_transform(y)

        if return_features:
            return self.pca_features

    def featureagg(
        self,
        n_clusters: int = 10,
        verbose: bool = False,
        return_features: bool = False,
        **kwargs,
    ):
        """Recursively merges pairs of clusters of features that minimally increases a given linkage distance.

        Args:
            n_clusters (int, optional): Number of clusters. Defaults to 10.
            verbose (bool, optional): _description_. Defaults to False.
            return_features (bool, optional): _description_. Defaults to False.
            **kwargs: Additional parameters to pass to FeatureAgglomeration.
        Returns:
            _type_: _description_
        """

        x, y = self.check_xinterp(verbose)

        fa = FeatureAgglomeration(n_clusters=n_clusters, **kwargs)
        self.fa_features = fa.fit_transform(y)

        if return_features:
            return self.fa_features

    def shapelet(self, labels=None, verbose=False, return_features=False, **kwargs):
        """Transforms a data set to identify shapelets. Only works for classification?

        Parameters
        ----------
        labels : np.array, optional, default=None
            target labels for shapelet transform.
        verbose : bool, optional, default=False
            print progress.
        return_features : bool, optional, default=False
            return features.
        **kwargs: dict
            additional parameters to pass to ShapeletTransform.

        Attributes
        ----------
        shapelet_features : np.array
            shapelet features.
        """

        x, y = self.check_xinterp(verbose)
        st = ShapeletTransform(**kwargs)
        self.shapelet_features = st.fit_transform(X=y, y=labels)
        if return_features:
            return self.shapelet_features

    def b_splines(self):
        """Not implemented yet."""
        pass

    def bow_features(self, **kwargs):
        """Not implemented yet."""
        bow = BagOfWords(**kwargs)
        pass

    def bop_features(self, **kwargs):
        """Not implemented yet."""
        bop = BagOfPatterns(**kwargs)
        pass

    def boss_features(self, **kwargs):
        """Not implemented yet."""
        boss = BOSS(**kwargs)
        pass

    def WEASEL_features(self, **kwargs):
        """Not implemented yet."""
        wsl = WEASEL(**kwargs)
        pass

    def ROCKET_features(self, **kwargs):
        """Not implemented yet."""
        rkt = ROCKET(**kwargs)
        pass

    def check_xinterp(self, verbose=False):
        """Check if interpolated data exists and return either interpolated data or raw data.

        Parameters
        ----------
        verbose : bool, optional, Default=False
            print progress.

        Returns
        -------
        x : np.array
            x values.
        y : np.array
            y values.
        """

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
