"""Plotting utilities for SpectraScope feature and model visualizations.

Provides three main classes:
- ``SpectraScopePlotting``: base class with shared axis/figure configuration.
- ``PlotSpectralFeatures``: visualize raw feature arrays (CWT, PCA, CDF, FFT, etc.).
- ``PlotModel``: overlay model-selected features on spectra for result interpretation.
"""

import warnings

import plotly.graph_objects as go
import numpy as np
from spectrascope.utils import find_nearest, func_unfix_width_reg
from PIL import Image
from matplotlib.colors import rgb2hex
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.fft import fftfreq
from pathlib import Path

__docformat__ = "numpy"

_STATIC_DIR = Path(__file__).resolve().parent.parent / "app" / "static"

# Icon per polynomial coefficient degree. This is the meaningful mapping: a
# feature's icon should show the basis function whose coefficient it holds.
POLY_ICONS = {0: "const", 1: "line", 2: "par", 3: "cubic"}

# Legacy mapping, icon per bin-count block (all features in the N=4 block drew
# "const", the N=5 block "line", and so on). Kept for reproducing older figures.
BLOCK_ICONS = {4: "const", 5: "line", 10: "par", 20: "cubic"}


def even_bin_indices(x, N):
    """Bin index of every point in ``x`` for ``N`` even-width bins.

    Reproduces ``SpectralFeatures.generate_bins`` exactly, including the
    truncating division and folding the top edge into the last bin.

    Parameters
    ----------
    x : array-like
        Axis the features were computed on.
    N : int
        Number of bins.

    Returns
    -------
    np.ndarray of int, shape (len(x),)
    """
    x = np.asarray(x, dtype=float)
    N = int(N)
    bin_width = (x.max() - x.min()) / N
    indices = np.clip(((x - x.min()) / bin_width).astype(int), 0, N - 1)
    if len(np.unique(indices)) != N:
        raise ValueError(
            f"an axis of {len(x)} points cannot fill {N} even-width bins; "
            "generate_bins would have skipped this N, so no feature block exists"
        )
    return indices


def poly_index_map(Narr=(4, 5, 10, 20), degree=3, trailing_max_feature=True):
    """Map each ``polynomial_featuresall`` column to the bin and coefficient it holds.

    ``polynomial_featuresall`` nests three loops: for each ``N`` in ``Narr``, for
    each of the ``N`` bins left to right, for each of the ``degree + 1``
    coefficients. ``np.polyfit`` returns coefficients highest-degree-first, so
    within a bin's group the order is ``a_degree ... a_0``. One trailing column
    (the x-position of the maximum intensity) is appended by
    ``polynomial_featuresall``, giving ``(degree + 1) * sum(Narr) + 1`` columns.

    Parameters
    ----------
    Narr : sequence of int, optional
        Bin counts, in the same order passed to ``polynomial_featuresall``.
    degree : int, optional
        Polynomial degree used for the fits.
    trailing_max_feature : bool, optional
        Whether the vector ends with the x-at-max column. True for
        ``polynomial_featuresall``, False for a bare ``polynomial_featuresN``.

    Returns
    -------
    dict
        ``{column index: (N, bin_index, coefficient_degree)}``, with the trailing
        x-at-max column mapped to ``None``.

    Example
    -------
    >>> poly_index_map()[18]
    (5, 0, 1)
    """
    index_map, i = {}, 0
    for N in Narr:
        for bin_index in range(int(N)):
            for k in range(degree + 1):
                index_map[i] = (int(N), bin_index, degree - k)
                i += 1
    if trailing_max_feature:
        index_map[i] = None
    return index_map


def _normalize_segments(segments, Narr):
    """Validate segment dicts and precompute their bin indices."""
    prepared = []
    for segment in segments:
        seg_x = np.asarray(segment["xaxis"], dtype=float)
        seg_y = np.asarray(segment["raw"], dtype=float)
        plot_x = np.asarray(segment.get("plot_x", seg_x), dtype=float)
        if not (len(seg_x) == len(seg_y) == len(plot_x)):
            raise ValueError(
                "segment 'xaxis', 'raw' and 'plot_x' must be the same length; got "
                f"{len(seg_x)}, {len(seg_y)}, {len(plot_x)}"
            )
        prepared.append(
            {
                "y": seg_y,
                "plot_x": plot_x,
                "bins": {int(N): even_bin_indices(seg_x, N) for N in Narr},
            }
        )
    return prepared


def locate_poly_features(
    selected_features,
    segments,
    Narr=(4, 5, 10, 20),
    degree=3,
    trailing_max_feature=True,
    icon_by="coefficient",
):
    """Resolve polynomial feature indices to positions on one or more spectra.

    Shared by the matplotlib and plotly plotters so both place icons identically.

    Each feature index is split into ``(segment, local index)`` by the
    per-spectrum feature count, so a multi-modal vector built by
    ``SpectraScope.combine_feature`` (a plain column concatenation) resolves its
    second half onto the second spectrum. The position of a coefficient feature
    is the mean of its bin's x values -- the point ``polynomial_featuresN``
    centers the fit on -- expressed in the target axes via ``plot_x``.

    Parameters
    ----------
    selected_features : array-like of int
        Column indices into the (possibly concatenated) polynomial feature vector.
    segments : list of dict
        One entry per concatenated spectrum, in feature-vector order:
        ``'xaxis'`` the axis the features were computed on (used for binning),
        ``'raw'`` its intensities, and optionally ``'plot_x'`` the x coordinates
        to draw at (defaults to ``'xaxis'``).
    Narr, degree, trailing_max_feature : optional
        Feature-layout parameters, see ``poly_index_map``.
    icon_by : {'coefficient', 'block'}, optional
        Choose the icon from the coefficient degree, or from the bin-count block
        to reproduce pre-fix figures.

    Returns
    -------
    list of dict
        One record per input feature, in input order:
        ``feature_id``, ``segment``, ``N``, ``bin``, ``coef_degree``, ``icon``
        (None where no icon image applies), ``center``, ``height``, and
        ``stack`` (how many earlier features share this position).

    Raises
    ------
    ValueError
        If any index falls outside the described layout, rather than silently
        dropping it.
    """
    if icon_by not in ("coefficient", "block"):
        raise ValueError(f"icon_by must be 'coefficient' or 'block', got {icon_by!r}")

    prepared = _normalize_segments(segments, Narr)
    index_map = poly_index_map(Narr, degree, trailing_max_feature)
    n_per_segment = len(index_map)

    records, unmapped, occupied = [], [], {}
    for feature_id in np.asarray(selected_features).ravel():
        feature_id = int(feature_id)
        segment_id, local = divmod(feature_id, n_per_segment)
        if feature_id < 0 or segment_id >= len(prepared) or local not in index_map:
            unmapped.append(feature_id)
            continue

        segment = prepared[segment_id]
        entry = index_map[local]
        if entry is None:
            # trailing feature: x-position of this spectrum's maximum
            N = bin_index = coef_degree = None
            center = float(segment["plot_x"][np.argmax(segment["y"])])
            icon = None
        else:
            N, bin_index, coef_degree = entry
            bin_mask = segment["bins"][N] == bin_index
            center = float(segment["plot_x"][bin_mask].mean())
            icon = (
                POLY_ICONS.get(coef_degree)
                if icon_by == "coefficient"
                else BLOCK_ICONS.get(N)
            )
        height = float(segment["y"][np.abs(segment["plot_x"] - center).argmin()])

        key = (segment_id, round(center, 6))
        stack = occupied.get(key, 0)
        occupied[key] = stack + 1

        records.append(
            {
                "feature_id": feature_id,
                "segment": segment_id,
                "N": N,
                "bin": bin_index,
                "coef_degree": coef_degree,
                "icon": icon,
                "center": center,
                "height": height,
                "stack": stack,
            }
        )

    if unmapped:
        raise ValueError(
            f"{len(unmapped)} feature index/indices fall outside a "
            f"{len(prepared)}-segment layout of {n_per_segment} features each "
            f"(valid range 0..{len(prepared) * n_per_segment - 1}): {unmapped}. "
            "For multi-modal features pass one 'segments' entry per spectrum; "
            "otherwise check Narr, degree and trailing_max_feature."
        )
    return records


class SpectraScopePlotting:
    """Base class for plotting spectral features."""

    def __init__(
        self,
        xaxis=None,
        xaxis_label="",
        yaxis_label="",
        twoD=False,
        name="",
        fig=None,
        row=1,
        col=1,
    ) -> None:
        """

        Parameters
        ----------
        xaxis : np.ndarray, optional
            X-axis values for plotting. Default None.
        xaxis_label : str, optional
            Label for the x-axis. Default "".
        yaxis_label : str, optional
            Label for the y-axis. Default "".
        twoD : bool, optional
            Reserved for future 2D plotting support. Default False.
        name : str, optional
            Trace name used in the plot legend. Default "".
        fig : plotly.graph_objects.Figure, optional
            Existing Plotly figure to add traces to. Default None.
        row : int, optional
            Subplot row index (1-based) when using make_subplots. Default 1.
        col : int, optional
            Subplot column index (1-based) when using make_subplots. Default 1.
        """
        self.xaxis_label = xaxis_label
        self.yaxis_label = yaxis_label
        self.fig = fig
        self.row = row
        self.col = col
        self.name = name
        self.xaxis = xaxis
        if twoD:
            pass


class PlotSpectralFeatures(SpectraScopePlotting):
    """Plotting class for spectral features."""

    def __init__(self, arr, **kwargs) -> None:
        """

        Parameters
        ----------
        arr : np.array or list
            feature array.
        """
        super().__init__(**kwargs)

        self.feature = arr
        if self.xaxis is None:
            self.xaxis = np.arange(len(self.feature))

    def get_xy(self):
        """Get the x and y values for plotting.

        Returns
        -------
        self.xaxis : (array-like)
            xaxis values
        self.feature : (array-like)
            feature values
        """

        return self.xaxis, self.feature

    def plot_raw_spec(self):
        """Plot the raw spectrum."""
        assert self.fig is not None, "please provide a figure for plotting"
        for yi in self.feature:
            self.fig.add_trace(
                go.Scatter(
                    x=self.xaxis,
                    y=yi,
                    mode="lines",
                    opacity=0.1,
                    showlegend=False,
                    line=dict(color="#7f7f7f"),
                ),
                row=self.row,
                col=self.col,
            )

        self.fig.add_trace(
            go.Scatter(
                x=self.xaxis, y=self.feature.mean(axis=0), mode="lines", name="average"
            ),
            row=self.row,
            col=self.col,
        )

        self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
        self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)

    def plot(self):
        """Not implemented yet."""

        pass

    def plot_pointwise_features(self, **kwargs):
        """Plot pointwise features.

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Scatter.
        """
        if self.fig is None:
            pass
        else:
            self.fig.add_trace(
                go.Scatter(x=self.xaxis, y=self.feature, **kwargs),
                row=self.row,
                col=self.col,
            )
            self.fig.update_xaxes(
                title_text=self.xaxis_label, row=self.row, col=self.col
            )
            self.fig.update_yaxes(
                title_text=self.yaxis_label, row=self.row, col=self.col
            )

    def plot_gaussian_peaks(self, **kwargs):
        """Summary of the method.

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Scatter.
        Returns
        -------
        np.array
            x values for plotting gaussian peaks.
        fitted : np.array
            y values for plotting gaussian peaks.
        """
        # TODO: This method is broken and unused (dead code). Issues:
        #  - `x` is undefined (never passed in or computed).
        #  - `func_unfix_width_reg` is undefined (not imported or defined).
        #  - `fitted` is undefined (never assigned).
        #  - First `add_trace` call passes positional args instead of a
        #    go.Scatter object, which is the wrong Plotly API.
        #  - Second `add_trace` passes `linestyle` and `color` kwargs that
        #    are matplotlib-style, not valid for go.Scatter.
        raise NotImplementedError("plot_gaussian_peaks is not yet implemented")

    def plot_cwt(self, **kwargs):
        """Plot continuous wavelet transform.
        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Heatmap.

        Returns
        -------
        np.array
            Reshaped feature array.
        """
        widths = len(self.feature) // len(self.xaxis)
        widths = int(widths)
        if len(self.feature) % len(self.xaxis) != 0:
            widths += 1
        temp = np.full((widths, len(self.xaxis)), np.nan)
        temp.flat[: len(self.feature)] = self.feature

        if self.fig is None:
            return temp
        else:
            self.fig.add_trace(go.Heatmap(z=temp, **kwargs), row=self.row, col=self.col)
            self.fig.update_xaxes(
                title_text=self.xaxis_label, row=self.row, col=self.col
            )
            self.fig.update_yaxes(
                title_text=self.yaxis_label, row=self.row, col=self.col
            )

    def plot_cdf(self, **kwargs):
        """Plot cumulative distribution function.
        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Scatter.

        Returns
        -------
        plot_x : np.array
            x values for plotting.
        plot_y : np.array
            y values for plotting.
        """
        if self.fig is None:
            plot_x, plot_y = self.get_xy()
            return plot_x, plot_y
        else:
            self.fig.add_trace(
                go.Scatter(x=self.xaxis, y=self.feature, **kwargs),
                row=self.row,
                col=self.col,
            )
            self.fig.update_xaxes(
                title_text=self.xaxis_label, row=self.row, col=self.col
            )
            self.fig.update_yaxes(
                title_text=self.yaxis_label, row=self.row, col=self.col
            )

    def plot_polynomial_featuresall(self, **kwargs):
        """Plot polynomial features.
        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Scatter.

        Returns
        -------
        plot_x : np.array
            x values for plotting.
        plot_y : np.array
            y values for plotting.
        """
        self.xaxis = np.arange(len(self.feature))
        if self.fig is None:
            plot_x, plot_y = self.get_xy()
            return plot_x[:-1], plot_y[:-1]
        else:
            self.fig.add_trace(
                go.Scatter(x=self.xaxis[:-1], y=self.feature[:-1], **kwargs),
                row=self.row,
                col=self.col,
            )
            self.fig.update_xaxes(
                title_text=self.xaxis_label, row=self.row, col=self.col
            )
            self.fig.update_yaxes(
                title_text=self.yaxis_label, row=self.row, col=self.col
            )

    def plot_pca(self, c=(0, 1), **kwargs):
        """Plot principal component features.
        Parameters
        ----------
        c : tuple, optional, default=(0, 1)
            selected components to plot. default first and second components.
        **kwargs : dict
            additional keyword arguments for go.Scatter.
        """
        c1, c2 = c
        if self.feature.shape[1] == 1:
            c1, c2 = 0, 0
        if self.fig is None:
            pass
        else:
            self.fig.add_trace(
                go.Scatter(x=self.feature[:, c1], y=self.feature[:, c2], **kwargs),
                row=self.row,
                col=self.col,
            )
            self.fig.update_xaxes(
                title_text=self.xaxis_label, row=self.row, col=self.col
            )
            self.fig.update_yaxes(
                title_text=self.yaxis_label, row=self.row, col=self.col
            )

    def plot_featureagg(self, c=(0, 1), **kwargs):
        """Plot feature agglomeration features.
        Uses plot_pca.

        Parameters
        ----------
        c : tuple, optional, default=(0, 1)
            selected components to plot. default first and second components.
        **kwargs : dict
            additional keyword arguments for go.Scatter.
        """
        self.plot_pca(c, **kwargs)

    def plot_tsne(self, c=(0, 1), **kwargs):
        """Plot t-distributed stochastic neighbor embedding features.
        Uses plot_pca.

        Parameters
        ----------
        c : tuple, optional, default=(0, 1)
            selected components to plot. default first and second components.
        **kwargs : dict
            additional keyword arguments for go.Scatter.
        """
        self.plot_pca(c, **kwargs)

    def plot_fft_features(self, **kwargs):
        """Plot fast fourier transform features.
        Needs original xaxis.

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Scatter.
        """
        from scipy.fft import fftfreq

        N = len(self.xaxis)
        de = np.diff(self.xaxis)[0]
        fft_freq = np.abs(fftfreq(N, de))[: N // 2]

        self.feature = np.abs(self.feature)
        if self.fig is None:
            pass
        else:
            self.fig.add_trace(
                go.Scatter(x=fft_freq, y=self.feature, **kwargs),
                row=self.row,
                col=self.col,
            )
            self.fig.update_xaxes(
                title_text=self.xaxis_label, row=self.row, col=self.col
            )
            self.fig.update_yaxes(
                title_text=self.yaxis_label, row=self.row, col=self.col
            )

    def plot_feature(self, feat):
        """Plot a feature.
        Parameters
        ----------
        feat : str
            feature to plot.
        """
        try:
            getattr(self, f"plot_{feat}")()
        except AttributeError:
            raise AttributeError(f"Feature {feat} not found.")


class PlotModel(SpectraScopePlotting):
    """Plotting class for model results."""

    def __init__(
        self,
        model=None,
        selected_features=None,
        raw=None,
        cmap_name="viridis",
        selected_vals=None,
        plot_importances=False,
        **kwargs,
    ) -> None:
        """

        Parameters
        ----------
        model : _type_, optional, default=None
            Fitted model object. Either a dict (LCEN) or sklearn model.
        selected_features : list or np.array, optional, default=None
            Indices of selected features to plot.
        raw : _type_, optional, default=None
            Raw data to plot. Often pointwise features.
        cmap_name : str, optional
            colormap for plotting. Defaults to "viridis".
        selected_vals : list or np.array, optional, default=None
            Values of selected features to plot. Defaults to None.
        plot_importances : bool, optional, Default=False
            Whether to plot feature importances as either line plot or color, or just plot locations of selected features. Defaults to False.
        **kwargs : dict
            additional keyword arguments for plotting.
        """
        super().__init__(**kwargs)
        self.model = model

        self.selected_features = selected_features
        self.plot_importances = plot_importances

        self.raw = raw
        if selected_vals is None:
            self.selected_vals = np.zeros_like(self.selected_features)
            self.show_scale = False
        else:
            self.selected_vals = selected_vals
            self.show_scale = True
        cmap = cm.get_cmap(cmap_name, len(self.selected_features))  # PiYG

        self.colors = []
        for i in range(cmap.N):
            rgba = cmap(i)
            self.colors.append(rgb2hex(rgba))

    def plot_polynomial_featuresall(
        self,
        sizex=100,
        sizey=10,
        segments=None,
        Narr=(4, 5, 10, 20),
        degree=3,
        trailing_max_feature=True,
        icon_by="coefficient",
    ):
        """Plot polynomial features selected features.

        Overlays a basis-function icon at the bin each selected coefficient was
        fit on. Index resolution is shared with
        ``plot_spectral_features_matplotlib`` via ``locate_poly_features``.

        Parameters
        ----------
        sizex : int, optional
            x size of polynomial annotation. Defaults to 100.
        sizey : int, optional
            y size of polynomial annotation. Defaults to 10.
        segments : list of dict, optional
            One entry per concatenated spectrum for multi-modal feature vectors,
            see ``locate_poly_features``. Default treats ``self.xaxis`` /
            ``self.raw`` as a single spectrum.
        Narr, degree, trailing_max_feature : optional
            Feature-layout parameters, see ``poly_index_map``.
        icon_by : {'coefficient', 'block'}, optional
            Icon from the coefficient degree, or from the bin-count block to
            reproduce pre-fix figures.

        Returns
        -------
        records : list of dict
            Located features, see ``locate_poly_features``.
        """

        # resolve indices before drawing, so a bad layout fails before the
        # figure is half-populated
        if segments is None:
            segments = [{"xaxis": self.xaxis, "raw": self.raw}]
        records = locate_poly_features(
            self.selected_features,
            segments,
            Narr=Narr,
            degree=degree,
            trailing_max_feature=trailing_max_feature,
            icon_by=icon_by,
        )

        self.fig.add_trace(
            go.Scatter(x=self.xaxis, y=self.raw, showlegend=False),
            row=self.row,
            col=self.col,
        )

        for record in records:
            # keep co-located features from hiding one another
            height = record["height"] + record["stack"] * sizey * 0.6

            if record["icon"] is None:
                # trailing x-at-max feature, or a coefficient above cubic that
                # has no icon image: label it instead
                text = (
                    "max"
                    if record["coef_degree"] is None
                    else f"a<sub>{record['coef_degree']}</sub>"
                )
                self.fig.add_annotation(
                    x=record["center"],
                    y=height,
                    text=text,
                    showarrow=False,
                    row=self.row,
                    col=self.col,
                )
                continue

            img_path = _STATIC_DIR / f"{record['icon']}.png"
            if not img_path.exists():
                warnings.warn(f"{img_path} not found.")
                continue
            im = Image.open(img_path)

            self.fig.add_layout_image(
                dict(
                    source=im,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=record["center"],
                    y=height,
                    sizex=sizex,
                    sizey=sizey,
                    # sizing="contain",
                    opacity=1,
                    layer="above",
                ),
                row=self.row,
                col=self.col,
            )
        self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
        self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)
        return records

    def plot_pointwise_features(self, **kwargs):
        """Plot pointwise features selected features.

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Scatter.
        """
        self.fig.add_trace(
            go.Scatter(x=self.xaxis, y=self.raw, showlegend=False, name="Raw Data"),
            row=self.row,
            col=self.col,
        )
        if self.plot_importances:
            self.fig.add_trace(
                go.Scatter(
                    x=self.xaxis,
                    y=self.selected_features,
                    name="Feature Importances",
                    **kwargs,
                ),
                secondary_y=True,
                row=self.row,
                col=self.col,
            )
        else:
            self.plot_lines()
        self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
        self.fig.update_yaxes(
            title_text=self.yaxis_label, row=self.row, col=self.col, secondary_y=False
        )

        self.fig.layout.yaxis2.update(showticklabels=False)

    def plot_cwt(self, wavelet="mexh", **kwargs):
        """Not implemented yet.

        Parameters
        ----------
        wavelet : str, optional, Default = "mexh"
            wavelet used in model fitting.
        """
        xaxis_label = self.xaxis_label or "Translation"
        yaxis_label = self.yaxis_label or "Scale"

        arr = kwargs["feat"]
        kwargs.pop("feat")
        if self.plot_importances:
            features = self.selected_features
        else:
            features = np.zeros(arr.shape[1], dtype=float)
            features[self.selected_features] = self.selected_vals

        cwt_plotter = PlotSpectralFeatures(
            features,
            fig=self.fig,
            xaxis=self.xaxis,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label,
            row=self.row,
            col=self.col,
        )
        cwt_plotter.plot_cwt(**kwargs)
        self.fig.update_xaxes(title_text=xaxis_label, row=self.row, col=self.col)
        self.fig.update_yaxes(title_text=yaxis_label, row=self.row, col=self.col)

    def plot_cdf(self, **kwargs):
        """Plot cumulative distribution function selected features.

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Scatter.

        """
        self.fig.add_trace(
            go.Scatter(x=self.xaxis, y=self.raw, showlegend=False),
            row=self.row,
            col=self.col,
        )
        if self.plot_importances:
            self.fig.add_trace(
                go.Scatter(
                    x=self.xaxis,
                    y=self.selected_features,
                    name="Feature Importances",
                    **kwargs,
                ),
                secondary_y=True,
                row=self.row,
                col=self.col,
            )
        else:
            self.plot_lines()
        self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
        self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)

    def plot_featureagg(self, **kwargs):
        """Plot feature agglomeration selected features.

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for PlotSpectralFeatures.plot_featureagg.
        """

        c = self.selected_features[:2]
        arr = kwargs["feat"]
        kwargs.pop("feat")
        featureagg_plotter = PlotSpectralFeatures(
            arr,
            fig=self.fig,
            xaxis=self.xaxis,
            xaxis_label=self.xaxis_label,
            yaxis_label=self.yaxis_label,
            row=self.row,
            col=self.col,
        )
        featureagg_plotter.plot_featureagg(c, **kwargs)

    def plot_pca(self, **kwargs):
        """Plot principal component analysis selected features.

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for PlotSpectralFeatures.plot_pca.
        """

        arr = kwargs["feat"]
        kwargs.pop("feat")
        c = self.selected_features[:2]
        pca_plotter = PlotSpectralFeatures(
            arr,
            fig=self.fig,
            xaxis=self.xaxis,
            xaxis_label=self.xaxis_label,
            yaxis_label=self.yaxis_label,
            row=self.row,
            col=self.col,
        )
        pca_plotter.plot_pca(c, **kwargs)

    def plot_tsne(self, **kwargs):
        """Plot t-distributed stochastic neighbor embedding selected features.

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for PlotSpectralFeatures.plot_tsne.
        """

        arr = kwargs["feat"]
        kwargs.pop("feat")
        c = self.selected_features[:2]
        tsne_plotter = PlotSpectralFeatures(
            arr,
            fig=self.fig,
            xaxis=self.xaxis,
            xaxis_label=self.xaxis_label,
            yaxis_label=self.yaxis_label,
            row=self.row,
            col=self.col,
        )
        tsne_plotter.plot_tsne(c, **kwargs)

    def plot_fft_features(self, **kwargs):
        """Plot fft features."""
        arr = kwargs["feat"]
        kwargs.pop("feat")

        N = len(self.xaxis)
        de = np.diff(self.xaxis)[0]
        fft_freq = np.abs(fftfreq(N, de))[: N // 2]
        xaxis_label = "Frequency"

        # Plot the fft
        self.fig.add_trace(
            go.Scatter(x=fft_freq, y=arr, **(kwargs | {"name": "fft"})),
            row=self.row,
            col=self.col,
        )
        self.fig.update_xaxes(title_text=xaxis_label, row=self.row, col=self.col)
        self.fig.update_yaxes(title_text="fft", row=self.row, col=self.col)

        # Plot the fft features
        y_name = (
            ""  # Feature Importances" if self.plot_importances else "Coefficient Value"
        )
        if self.plot_importances:
            # Plot the selected features
            self.fig.add_trace(
                go.Scatter(
                    x=fft_freq,
                    y=self.selected_features,
                    **(kwargs | {"name": "Feature Importances"}),
                ),
                row=self.row,
                col=self.col,
            )
        else:
            # Normalize the selected values to the range [0, 1]
            norm = mcolors.Normalize(
                vmin=min(self.selected_vals), vmax=max(self.selected_vals)
            )
            cmap = cm.get_cmap("viridis")
            colors = [mcolors.rgb2hex(cmap(norm(val))) for val in self.selected_vals]
            y_min, y_max = np.min(arr), np.max(arr)

            for i, x in enumerate(fft_freq[self.selected_features]):
                curr_color = (
                    colors[i] if self.show_scale else mcolors.rgb2hex(cmap(0.5))
                )
                curr_val = self.selected_vals[i]
                curr_feature = self.selected_features[i]
                self.fig.add_trace(
                    go.Scatter(
                        x=[x, x],
                        y=[y_min, y_max],
                        mode="lines",
                        line=dict(color=curr_color, width=3),
                        hovertemplate=f"Feature ID: {curr_feature}<br>Value: {curr_val}",
                        showlegend=False,
                    )
                )

            self.fig.update_layout(hovermode="x")
        self.fig.update_xaxes(title_text=xaxis_label, row=self.row, col=self.col)
        self.fig.update_yaxes(title_text=y_name, row=self.row, col=self.col)

    def plot_gaussian_peaks(self):
        """Not implemented yet."""
        pass

    def plot_lines(self, **kwargs):
        """Plot lines at the location of selected features

        Parameters
        ----------
        **kwargs : dict
            additional keyword arguments for go.Scatter.

        """

        # Normalize the selected values to the range [0, 1]
        norm = mcolors.Normalize(
            vmin=min(self.selected_vals), vmax=max(self.selected_vals)
        )
        cmap = cm.get_cmap("viridis")
        colors = [mcolors.rgb2hex(cmap(norm(val))) for val in self.selected_vals]
        for i, x in enumerate(self.xaxis[self.selected_features]):
            self.fig.add_vline(
                x=x,
                line=dict(
                    color=colors[i] if self.show_scale else mcolors.rgb2hex(cmap(0.5)),
                    width=3,
                ),
            )

        self.fig.update_layout(hovermode="x")


from spectrascope.Features import SpectralFeatures
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def _load_icon(icon, img_dir=None, resize=True, max_size=128):
    """Open an icon image, optionally downsampled so overlays stay light."""
    img_path = Path(img_dir if img_dir is not None else _STATIC_DIR) / f"{icon}.png"
    if not img_path.exists():
        warnings.warn(f"{img_path} not found.")
        return None
    im = Image.open(img_path)
    if not resize:
        return im
    width, height = im.size
    scale = min(max_size / width, max_size / height, 1.0)
    return np.asarray(
        im.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
    )


def plot_spectral_features_matplotlib(
    xaxis,
    raw,
    selected_features,
    ax=None,
    zoom=0.1,
    plot_raw=True,
    print_centers=False,
    plot=True,
    resize=True,
    segments=None,
    Narr=(4, 5, 10, 20),
    degree=3,
    trailing_max_feature=True,
    icon_by="coefficient",
    img_dir=None,
):
    """
    Plot spectral features using matplotlib with annotated representative images.

    Each selected polynomial feature is drawn as the basis-function icon for the
    coefficient it holds, positioned at the bin that coefficient was fit on.
    Index resolution is shared with ``PlotModel.plot_polynomial_featuresall`` via
    ``locate_poly_features``.

    Parameters
    ----------
    xaxis : np.ndarray
        Array representing the x-axis values of the spectrum (e.g., energy or wavelength).
        Used for drawing the spectrum line, and for locating features when
        ``segments`` is not given.
    raw : np.ndarray
        The raw intensity values of the spectrum corresponding to xaxis.
    selected_features : list or np.ndarray
        List of indices of selected features to annotate on the plot.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis object to plot on. If None, a new axis will be created.
    zoom : float, optional
        Zoom factor for displaying the annotated images. Default is 0.1.
    plot_raw : bool, optional
        Whether to plot the raw spectrum as a black line. Default is True.
    print_centers : bool, optional
        If True, prints the center x-values and corresponding selected feature indices.
    plot : bool, optional
        Whether to draw the icons. False locates features without drawing.
    resize : bool, optional
        Downsample icon images before overlaying them. Default True.
    segments : list of dict, optional
        One entry per concatenated spectrum, required for multi-modal feature
        vectors built by ``SpectraScope.combine_feature``. See
        ``locate_poly_features``. Default treats ``xaxis`` / ``raw`` as a single
        spectrum.
    Narr, degree, trailing_max_feature : optional
        Feature-layout parameters, see ``poly_index_map``.
    icon_by : {'coefficient', 'block'}, optional
        Icon from the coefficient degree, or from the bin-count block to
        reproduce pre-fix figures.
    img_dir : str or pathlib.Path, optional
        Directory holding the icon PNGs. Defaults to the packaged app/static.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object containing the generated plot.
    centers : np.ndarray, shape (len(selected_features), 2)
        ``(feature_id, x_center)`` pairs, one row per input feature in input
        order, so positional indexing stays aligned with ``selected_features``.

    Raises
    ------
    ValueError
        If a feature index falls outside the described layout.

    Notes
    -----
    A coefficient's position is the mean of its bin's x values, the point
    ``polynomial_featuresN`` centers the fit on. Features sharing a position
    (different coefficients of one bin) are staggered vertically. The trailing
    x-at-max feature, and any coefficient above cubic, have no icon image and
    are drawn as a marker or text label instead.

    Example
    -------
    >>> ax, centers = plot_spectral_features_matplotlib(
    ...     xaxis, spectrum, selected_features, ax=ax, zoom=0.03
    ... )
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the raw spectrum
    if plot_raw:
        ax.plot(xaxis, raw, color="black", lw=1)

    if segments is None:
        segments = [{"xaxis": xaxis, "raw": raw}]
    records = locate_poly_features(
        selected_features,
        segments,
        Narr=Narr,
        degree=degree,
        trailing_max_feature=trailing_max_feature,
        icon_by=icon_by,
    )

    spread = np.ptp(np.asarray(raw, dtype=float))
    for record in records:
        center, height = record["center"], record["height"]
        if print_centers:
            print(center, record["feature_id"])
        if not plot:
            continue

        # keep co-located features (different coefficients of one bin) visible
        height += record["stack"] * 0.08 * spread

        if record["icon"] is None:
            if record["coef_degree"] is None:
                ax.plot(center, height, marker="*", markersize=14, color="black", zorder=5)
            else:
                ax.annotate(
                    f"$a_{record['coef_degree']}$",
                    (center, height),
                    ha="center",
                    va="center",
                )
            continue

        im = _load_icon(record["icon"], img_dir=img_dir, resize=resize)
        if im is None:
            continue
        try:
            ax.add_artist(
                AnnotationBbox(OffsetImage(im, zoom=zoom), (center, height), frameon=False)
            )
        except ValueError as e:
            warnings.warn(
                f"{e}\n"
                "Image size may be too large. Try reducing zoom, setting plot_raw=True, "
                "or plotting other spectra on the axis first."
            )

    centers = np.array(
        [[record["feature_id"], record["center"]] for record in records], dtype=float
    ).reshape(-1, 2)
    return ax, centers
