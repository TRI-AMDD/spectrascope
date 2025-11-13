import plotly.graph_objects as go
import numpy as np
from spectrascope.utils import *
from PIL import Image
from matplotlib.colors import rgb2hex
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.fft import fftfreq

__docformat__ = "numpy"


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
        xaxis : _type_, optional
            _description_, by default None
        xaxis_label : str, optional
            _description_, by default ""
        yaxis_label : str, optional
            _description_, by default ""
        twoD : bool, optional
            _description_, by default False
        name : str, optional
            _description_, by default ""
        fig : _type_, optional
            _description_, by default None
        row : int, optional
            _description_, by default 1
        col : int, optional
            _description_, by default 1
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
        if self.fig is None:
            pass
        else:
            pass
        self.fig.add_trace(
            self.xaxis,
            func_unfix_width_reg(x, *self.feature),
            label="Reconstructed spectrum",
        )
        for i in range(0, len(self.feature), 3):
            ctr = self.feature[i]
            amp = self.feature[i + 1]
            wid = self.feature[i + 2]
            yfit = amp * np.exp(-((x - ctr) ** 2 / (2 * (wid / 2.355) ** 2)))
            self.fig.add_trace(
                go.Scatter(self.xaxis, yfit, linestyle="-.", color="grey", **kwargs)
            )

        fitted = np.array(fitted).ravel()
        return np.arange(len(fitted)), fitted

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

    def plot_polynomial_featuresall(self, sizex=100, sizey=10):
        """Plot polynomial features selected features.

        Parameters
        ----------
        sizex : int, optional
            x size of polynomial annotation. Defaults to 100.
        sizey : int, optional
            y size of polynomial annotation. Defaults to 10.
        """

        self.fig.add_trace(
            go.Scatter(x=self.xaxis, y=self.raw, showlegend=False),
            row=self.row,
            col=self.col,
        )
        from spectrascope.features import SpectralFeatures

        temp = SpectralFeatures(self.xaxis, self.raw)

        temp.generate_bins(4)
        temp.generate_bins(5)
        temp.generate_bins(10)
        temp.generate_bins(20)

        # plot_key = {}
        for selected_id in self.selected_features:
            if selected_id < 16:
                bin_label = "bin_indices_4"
                poly_label = "const"
                search_party = np.linspace(0, 16, 4)
            elif selected_id >= 16 and selected_id < 36:
                bin_label = "bin_indices_5"
                poly_label = "line"
                search_party = np.linspace(16, 36, 5)
            elif selected_id >= 36 and selected_id < 76:
                bin_label = "bin_indices_10"
                poly_label = "par"
                search_party = np.linspace(36, 76, 10)
            elif selected_id >= 76:
                bin_label = "bin_indices_20"
                poly_label = "cubic"
                search_party = np.linspace(76, 156, 20)
            located_bin = np.argwhere(selected_id < search_party)[0][0]
            center = np.median(self.xaxis[getattr(temp, bin_label) == located_bin])
            height_id = find_nearest(self.xaxis, center)
            height = self.raw[height_id]
            # plot_key[selected_id] = (center, poly_label, height)
            im = Image.open(f"../app/static/{poly_label}.png")

            self.fig.add_layout_image(
                dict(
                    source=im,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=center,
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

            # self.fig.update_layout(coloraxis_colorbar=dict(title="Coefficient or Importance"))
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

        # self.fig.update_layout(coloraxis_colorbar=dict(title="Coefficient or Importance"))
        self.fig.update_layout(hovermode="x")

        # self.fig.layout.yaxis2.update(showticklabels=False)

        # customdata=self.xaxis[self.selected_features], hovertemplate="x:%{customdata[0]}"


from spectrascope.features import SpectralFeatures
import os 
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_spectral_features_matplotlib(xaxis, raw, selected_features, ax=None,zoom=0.1,
                                      plot_raw=True, print_centers=False):
    """
    Plot spectral features using matplotlib with annotated representative images.

    Parameters
    ----------
    xaxis : np.ndarray
        Array representing the x-axis values of the spectrum (e.g., energy or wavelength).
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
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object containing the generated plot.

    Notes
    -----
    This function overlays annotated images (e.g., icons of feature types) at the
    bin centers of selected spectral features, using images from static files.
    The images corresponding to feature types are matched using a pre-defined mapping.
    The function assumes the existence of static images for each feature type in the
    '../app/static/' directory.

    Example
    -------
    >>> plot_spectral_features_matplotlib(xaxis, spectrum, selected_features, ax=ax, zoom=0.03)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the raw spectrum
    if plot_raw:
        ax.plot(xaxis, raw, color='black', lw=1)

    # Generate binned features
    temp = SpectralFeatures(xaxis, raw)
    for bins in [4, 5, 10, 20]:
        temp.generate_bins(bins)

    # Feature ID config
    feature_config = [
        (0, 16,  'bin_indices_4',  'const', 4),
        (16, 36, 'bin_indices_5',  'line',  5),
        (36, 76, 'bin_indices_10', 'par',   10),
        (76, 156, 'bin_indices_20', 'cubic', 20),
    ]

    def find_nearest(array, value):
        return np.abs(array - value).argmin()
    try: 
        centers_used = []
        for selected_id in selected_features:
            for id_min, id_max, bin_label, poly_label, num_bins in feature_config:
                if id_min <= selected_id < id_max:
                    search_bins = np.linspace(id_min, id_max, num_bins)
                    located_bin = np.argwhere(selected_id < search_bins)[0][0]
                    bin_mask = getattr(temp, bin_label) == located_bin
                    center = np.median(xaxis[bin_mask])
                    height_id = find_nearest(xaxis, center)
                    height = raw[height_id]

                    # Load image
                    img_path = f"../app/static/{poly_label}.png"
                    if not os.path.exists(img_path):
                        print(f"Warning: {img_path} not found.")
                        continue
                    im = Image.open(img_path)

                    # Convert image to matplotlib offsetbox
                    imagebox = OffsetImage(im, zoom=zoom)
                    if center in centers_used: 
                        add_to_height = 1  # slight offset to avoid overlap
                    else: 
                        add_to_height = 0 
                    ab = AnnotationBbox(imagebox, (center, height + add_to_height), frameon=False)
                    centers_used.append(center)
                    if print_centers: 
                        print(center, selected_id)

                    ax.add_artist(ab)
                    break  # stop after first matching range
    except ValueError as e:
        print(e) 
        print('Warning : Image size too large. You probably have to plot something else on this axis first.')
        print('Try reducing the zoom, set plot_raw=True, or plotting other spectra on the axis first')
    return ax