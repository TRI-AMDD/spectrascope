

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 
from spectrascope.utils import * 
from PIL import Image
from matplotlib.colors import rgb2hex
import matplotlib.cm as cm  


class ss_plotting: 
    def __init__(self, xaxis=None, xaxis_label='', yaxis_label='', 
                 twoD = False, name='', fig=None, row=1, col=1) -> None:

        self.xaxis_label = xaxis_label
        self.yaxis_label = yaxis_label
        self.fig = fig 
        self.row = row
        self.col = col
        self.name = name 
        self.xaxis = xaxis
        if twoD: 
            pass

class plot_spectral_features(ss_plotting):
    def __init__(self, arr, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.feature = arr
        if self.xaxis is None: 
            self.xaxis = np.arange(len(self.feature))

    
    def get_xy(self): 
        return self.xaxis, self.feature
    
    def plot_raw_spec(self): 
        assert self.fig is not None , "please provide a figure for plotting"
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
            go.Scatter(x=self.xaxis, y=self.feature.mean(axis=0), mode="lines", name="average"), row=self.row, col=self.col
        )

        self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
        self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)
    
    def plot(self):
        pass
    
    def plot_pointwise_features(self, **kwargs): 
        if self.fig is None: 
            pass 
        else: 
            self.fig.add_trace(go.Scatter(x=self.xaxis, y=self.feature, **kwargs), row=self.row, col=self.col)
            self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
            self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)
    
    def plot_gaussian_peaks(self, **kwargs):
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
            self.fig.add_trace(self.xaxis, yfit, linestyle="-.", color="grey", **kwargs)

        fitted = np.array(fitted).ravel()
        return np.arange(len(fitted)), fitted

    def plot_cwt(self, **kwargs):
        widths = len(self.feature) / len(self.xaxis)
        widths = int(widths)
        temp = self.feature.reshape(widths, len(self.xaxis))
        
        if self.fig is None:
            return temp
        else:
            self.fig.add_trace(go.Heatmap(z=temp, **kwargs), row=self.row, col=self.col)
            self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
            self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)

    def plot_cdf(self, **kwargs):

        if self.fig is None:
            plot_x, plot_y = self.get_xy()
            return plot_x, plot_y 
        else:  
            self.fig.add_trace(go.Scatter(x=self.xaxis, y=self.feature, **kwargs), row=self.row, col=self.col)
            self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
            self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)

    def plot_polynomial_featuresall(self, **kwargs):
        self.xaxis = np.arange(len(self.feature))
        if self.fig is None: 
            plot_x, plot_y = self.get_xy()
            return plot_x[:-1], plot_y[:-1]
        else: 
            self.fig.add_trace(go.Scatter(x=self.xaxis[:-1],y=self.feature[:-1], **kwargs), row=self.row, col=self.col)
            self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
            self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)

    def plot_pca(self, c = (0,1), **kwargs):
        c1, c2 = c  
        if self.fig is None: 
            pass
        else: 
            self.fig.add_trace(go.Scatter(x=self.feature[:,c1], y=self.feature[:,c2], **kwargs), row=self.row, col=self.col)
            self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
            self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)

    def plot_featureagg(self, c=(0,1), **kwargs): 
        self.plot_pca(c, **kwargs)
        
    def plot_tsne(self, c=(0,1), **kwargs):
        self.plot_pca(c, **kwargs)
        
    def plot_fft_features(self, **kwargs): 
        " needs original xaxis" 
        from scipy.fft import fftfreq
        N = len(self.xaxis)
        de = np.diff(self.xaxis)[0]
        fft_freq = np.abs(fftfreq(N, de))[: N // 2]
        
        self.feature = np.abs(self.feature)
        if self.fig is None: 
            pass 
        else: 
            self.fig.add_trace(go.Scatter(x=fft_freq, y = self.feature, **kwargs), row=self.row, col=self.col)
            self.fig.update_xaxes(title_text=self.xaxis_label, row=self.row, col=self.col)
            self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)

class plot_model(ss_plotting):
    def __init__(self, model=None, selected_features=None, raw=None, cmap_name='viridis', selected_vals=None, plot_importances=False, **kwargs) -> None:
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
        cmap = cm.get_cmap(cmap_name, len(self.selected_features))    # PiYG

        self.colors = []
        for i in range(cmap.N):
            rgba = cmap(i)
            self.colors.append(rgb2hex(rgba))
        
    def plot_polynomial_featuresall(self): 
        
        self.fig.add_trace(go.Scatter(x=self.xaxis, y=self.raw), row=self.row, col=self.col)
        for selected_id in self.selected_features: 
    
            if selected_id < 16:
                poly_label = 'const' 
            elif selected_id >= 16 and selected_id < 36:
                selected_id -= 16
                poly_label = 'line'
            elif selected_id >= 36 and selected_id < 76: 
                selected_id -= 36
                poly_label = 'par'
            elif selected_id >= 76: 
                selected_id -=76
                poly_label = 'cubic'
            
            center = self.xaxis[selected_id]
            height = self.raw[selected_id]
            im  = Image.open(f'../app/static/{poly_label}.png')

            self.fig.add_layout_image(
                dict(
                    source=im,
                    xref="x",
                    yref="y",
                    xanchor='right',
                    x=center,
                    y=height,
                    sizex=2 ,
                    sizey=2,
                    sizing="contain", 
                    opacity=1,
                    layer="below")
            )
        self.fig.update_xaxes(title_text=self.xaxis_label)
        self.fig.update_yaxes(title_text=self.yaxis_label)
        
            
    def plot_pointwise_features(self, **kwargs): 

        self.fig.add_trace(go.Scatter(x=self.xaxis, y=self.raw, showlegend=False, name='Raw Data'), row=self.row, col=self.col)
        if self.plot_importances: 
            self.fig.add_trace(go.Scatter(x=self.xaxis, y=self.selected_features, name='Feature Importances'),  secondary_y=True, row=self.row, col=self.col)
        else: 
            self.plot_lines()
        self.fig.update_xaxes(title_text=self.xaxis_label)
        self.fig.update_yaxes(title_text=self.yaxis_label)
        
        self.fig.layout.yaxis2.update(showticklabels=False)

    
    def plot_cwt(self, wavelet='mexh'):
        pass
    
    def plot_cdf(self, **kwargs): 
        
        self.fig.add_trace(go.Scatter(x=self.xaxis, y=self.raw, showlegend=False), row=self.row, col=self.col)
        if self.plot_importances: 
            self.fig.add_trace(go.Scatter(x=self.xaxis, y=self.selected_features, name='Feature Importances'),  secondary_y=True, row=self.row, col=self.col)
        else: 
            self.plot_lines()
        self.fig.update_xaxes(title_text=self.xaxis_label)
        self.fig.update_yaxes(title_text=self.yaxis_label) 
    
    def plot_featureagg(self, **kwargs):
        c = self.selected_features[:2]
        arr = kwargs["feat"]
        kwargs.pop("feat")
        featureagg_plotter = plot_spectral_features(arr, fig=self.fig, xaxis=self.xaxis, xaxis_label=self.xaxis_label, yaxis_label=self.yaxis_label, row=self.row, col=self.col)
        featureagg_plotter.plot_featureagg(c, **kwargs)
    
    def plot_pca(self, **kwargs):
        arr = kwargs["feat"]
        kwargs.pop("feat")
        c = self.selected_features[:2]
        pca_plotter = plot_spectral_features(arr, fig=self.fig, xaxis=self.xaxis, xaxis_label=self.xaxis_label, yaxis_label=self.yaxis_label, row=self.row, col=self.col)
        pca_plotter.plot_pca(c, **kwargs)
        
    def plot_tsne(self, **kwargs):
        arr = kwargs["feat"]
        kwargs.pop("feat")
        c = self.selected_features[:2]
        tsne_plotter = plot_spectral_features(arr, fig=self.fig, xaxis=self.xaxis, xaxis_label=self.xaxis_label, yaxis_label=self.yaxis_label, row=self.row, col=self.col)
        tsne_plotter.plot_tsne(c, **kwargs)
    
    def plot_fft_features(self):
        pass 
    
    def plot_gaussian_peaks(self):
        pass 
         
    def plot_lines(self, **kwargs): 
        kwargs = {'marker_size':15,'marker_symbol':'line-ns', 'marker_line_width':2,'mode':'markers', 'hoverinfo':'none','marker':dict(color=self.selected_vals, colorscale='Viridis', showscale=self.show_scale, line=dict(width=2,
                                        color=self.selected_vals, colorscale='Viridis'))}
        for i in np.arange(0, np.max(self.raw), 0.1):
            self.fig.add_trace(go.Scatter(x=self.xaxis[self.selected_features], y = i*np.ones_like(self.selected_features), showlegend=False, name='Selected Features',
                                    **kwargs), row=1, col=1, secondary_y=True)
        self.fig.update_layout(coloraxis_colorbar=dict(title="Coefficient or Importance"))
        self.fig.update_layout(hovermode="x")
        # fig.layout.yaxis2.update(showticklabels=False)

        #customdata=self.xaxis[self.selected_features], hovertemplate="x:%{customdata[0]}"
