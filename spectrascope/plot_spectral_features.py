

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 
from spectrascope.utils import * 

class plot_spectral_features:
    def __init__(self, arr, feature_name='', xaxis=None, xaxis_label='', yaxis_label='', 
                 twoD = False, fig=None, row=1, col=1, **kwargs) -> None:
        self.feature_name = feature_name
        
        if twoD: 
            pass 
        self.feature = arr
        
        if xaxis is None: 
            self.xaxis = np.arange(len(self.feature))
        else: 
            self.xaxis = xaxis 
        self.xaxis_label = xaxis_label
        self.yaxis_label = yaxis_label
        self.fig = fig 
        self.row = row
        self.col = col

    
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
        
    def plot_gaussian_peaks(self, **kwargs):
        if self.fig is None: 
            pass 
        else: 
            pass 
        ax.plot(
            self.xaxis,
            func_unfix_width_reg(x, *self.feature),
            label="Reconstructed spectrum",
        )
        for i in range(0, len(self.feature), 3):
            ctr = self.feature[i]
            amp = self.feature[i + 1]
            wid = self.feature[i + 2]
            yfit = amp * np.exp(-((x - ctr) ** 2 / (2 * (wid / 2.355) ** 2)))
            ax.plot(self.xaxis, yfit, linestyle="-.", color="grey", **kwargs)

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

    def plot_cdf(self, **kwargs):

        if self.fig is None:
            plot_x, plot_y = self.get_xy()
            return plot_x, plot_y 
        else:  
            self.fig.add_trace(go.Scatter(x=self.xaxis, y=self.feature, **kwargs), row=self.row, col=self.col)

    def plot_polynomial_featuresall(self, **kwargs):
        self.xaxis = np.arange(len(self.feature))
        if self.fig is None: 
            plot_x, plot_y = self.get_xy()
            return plot_x[:-1], plot_y[:-1]
        else: 
            self.fig.add_trace(go.Scatter(x=self.xaxis[:-1],y=self.feature[:-1], **kwargs), row=self.row, col=self.col)

    def plot_pca(self, c = (0,1), **kwargs):
        c1, c2 = c  
        if self.fig is None: 
            pass
        else: 
            self.fig.add_trace(go.Scatter(x=self.feature[:,c1], y=self.feature[:,c2], **kwargs), row=self.row, col=self.col)
            
    def plot_featureagg(self, c=(0,1), **kwargs): 
        c1, c2 = c  
        if self.fig is None: 
            pass 
        else: 
            self.fig.add_trace(go.Scatter(x=self.feature[:,c1], y=self.feature[:,c2], **kwargs), row=self.row, col=self.col)

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
    
    