

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 

class plot_spectral_features:
    def __init__(self, arr, feature_name='', xaxis=None, xaxis_label='', yaxis_label='', twoD = False, **kwargs) -> None:
        self.feature_name = feature_name
        
        if twoD == False: 
            self.feature = arr.ravel()
        
        if xaxis is None: 
            self.xaxis = np.arange(len(self.feature))
        else: 
            self.xaxis = xaxis 
        self.xaxis_label = xaxis_label
        self.yaxis_label = yaxis_label

    
    def get_xy(self): 
        return self.xaxis, self.feature
    
    def _plot(self, ax=None, fig=None):
        if ax == None: 
            pass 
        else: 
            # add to a specific axis or figure 
            pass  
        
    
    