"""
Spectrascope is a python package for feature generation, model training, and feature downselection of
spectral or array-like data.
"""

from spectrascope.SpectraScope import SpectraScope
from spectrascope.Features import SpectralFeatures
from spectrascope.SSPlotting import (
    PlotSpectralFeatures,
    PlotModel,
    plot_spectral_features_matplotlib,
    locate_poly_features,
    poly_index_map,
    even_bin_indices,
    POLY_ICONS,
    BLOCK_ICONS,
)
from spectrascope.SSClassification import BinaryClassifications

__all__ = [
    "SpectraScope",
    "SpectralFeatures",
    "PlotSpectralFeatures",
    "PlotModel",
    "plot_spectral_features_matplotlib",
    "locate_poly_features",
    "poly_index_map",
    "POLY_ICONS",
    "BLOCK_ICONS",
    "BinaryClassifications",
]
