"""
Spectrascope is a python package for feature generation, model training, and feature downselection of
spectral or array-like data.
"""

from spectrascope.SpectraScope import SpectraScope
from spectrascope.Features import SpectralFeatures
from spectrascope.SSPlotting import PlotSpectralFeatures, PlotModel, plot_spectral_features_matplotlib
from spectrascope.SSClassification import BinaryClassifications

__all__ = [
    "SpectraScope",
    "SpectralFeatures",
    "PlotSpectralFeatures",
    "PlotModel",
    "plot_spectral_features_matplotlib",
    "BinaryClassifications",
]
