import sys
import os 
import pathlib 
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from spectrascope.features import *


def test_spectral_features_class(): 
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    
    assert len(temp.energies) == len(temp.intensities)
    assert type(temp.energies) == type(temp.intensities) == np.ndarray
    
def test_pointwise(): 
    x = np.linspace(4970, 5022, 100)
    y = np.random.randint(200, size=(100,))
    temp = spectral_features(x, y)
    
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    
    xrange = np.linspace(4969.0, 5021.0, 200)
    assert len(temp.energies) == len(temp.intensities)
    
    temp.pointwise_features(xrange)
    
    assert hasattr(temp, 'xinterp')
    assert hasattr(temp, 'yinterp')
    
    assert len(temp.xinterp) == len(temp.yinterp) == 200
    
    xrange = np.linspace(4969.0, 5021.0, 300)
    temp.pointwise_features(xrange)
    assert len(temp.xinterp) == len(temp.yinterp) == 300
    
    xrange = np.linspace(4969.0, 5021.0, 100)
    temp.pointwise_features(xrange)
    assert len(temp.xinterp) == len(temp.yinterp) == 100
    
    

def test_polynomial():
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x,y)
    
    for N in [1,7,20]:
        for degree in [1, 3,5]: 
            temp.polynomial_featuresN(N, degree)
            assert hasattr(temp, f'polyfeatures_{N}_{degree}deg')
            featuresN = getattr(temp, f'polyfeatures_{N}_{degree}deg')
            assert featuresN.shape == np.zeros([N, degree+1]).shape 
            
    temp.polynomial_featuresN(7, 3, x=x[10:60], y=y[10:60])
    assert hasattr(temp, f'polyfeatures_{round(x[10])}_{round(x[60])}_{7}_{3}deg')
    assert featuresN.shape == np.zeros([N, degree+1]).shape 

def test_bins(): 
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    
    for N in [1, 5,20]: 
        temp.generate_bins(N)
        assert hasattr(temp, f'bin_indices_{N}')
        bins_generated = getattr(temp, f'bin_indices_{N}')
        assert np.max(bins_generated) == N - 1 
        assert np.min(bins_generated) == 0 
        
        assert (np.unique(bins_generated) == np.arange(N)).all()
        
    temp.generate_bins(5, x = x[:55], y = y[:55])
    minx = np.min(x)
    maxx = x[54]
    assert hasattr(temp, f'bin_indices_{round(minx)}_{round(maxx)}_{5}')
    

def test_polynomial_features_all(): 
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    temp.polynomial_featuresall()
    assert hasattr(temp, 'polyfeatures_all')
    featuresall = getattr(temp, 'polyfeatures_all')
    assert featuresall.shape == (157,) 
    
    temp.polynomial_featuresall(Narr = [3,1,4])
    assert hasattr(temp, 'polyfeatures_all')
    featuresall = getattr(temp, 'polyfeatures_all')
    assert featuresall.shape == (3 * 4 + 1 * 4 + 4 * 4 + 1,) 

    temp.polynomial_featuresall(degree=4, Narr = [3,10,7])
    assert hasattr(temp, 'polyfeatures_all')
    featuresall = getattr(temp, 'polyfeatures_all')
    assert featuresall.shape == (3 * 5 + 10 * 5 + 7 * 5 + 1,)     
    
    
def test_uneven(): 
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    
    eV_splits = [20,50]
    N_dict={0:10,1:2, 2:2}
    degree_dict={0:4,1:3,2:3}
    
    temp.uneven_features(eV_splits, N_dict, degree_dict, plot=False)
    idx_arr = [find_nearest(temp.energies, temp.energies.min() + eV) for eV in eV_splits]

    for i, idx in enumerate(idx_arr): 
        N = N_dict[i]
        degree = degree_dict[i]
        if i == 0: 
            start, stop = 0, idx 
            tempx, tempy = x[start:stop], y[start:stop]
            assert hasattr(temp, f'polyfeatures_{round(tempx.min())}_{round(tempx.max())}_{N}_{degree}deg')
        elif i < len(idx_arr): 
            start, stop = idx_arr[i-1],idx
            tempx, tempy = x[start:stop], y[start:stop]
            assert hasattr(temp, f'polyfeatures_{round(tempx.min())}_{round(tempx.max())}_{N}_{degree}deg')
    N = N_dict[i + 1]
    degree = degree_dict[i+1]
    start, stop = idx, -1 
    tempx, tempy = x[start:], y[start:]
    assert hasattr(temp, f'polyfeatures_{round(tempx.min())}_{round(tempx.max())}_{N}_{degree}deg')
        
    temp.uneven_features(eV_splits=[20], N_dict={0:10,1:5}, degree_dict={0:4,1:3}, plot=False)

        
def test_cdf(): 
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    temp.cdf()
    assert hasattr(temp, 'cdf')
    assert len(temp.cdf) == len(temp.intensities)

def test_cwt(): 
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    temp.cwt()
    assert hasattr(temp, 'cwt_coefs')
    assert hasattr(temp, 'cwt_frequencies')
    
    assert temp.cwt_coefs.shape[0] == temp.cwt_frequencies.shape[0]
    assert temp.cwt_coefs.shape == (10,100)
    
    temp.cwt(widths=np.arange(1,16))
    assert temp.cwt_coefs.shape[0] == temp.cwt_frequencies.shape[0]
    assert temp.cwt_coefs.shape == (15,100) 
    
    temp.pointwise_features(xrange=np.linspace(x.min(), x.max(), 200))
    temp.cwt(widths=np.arange(1,16))
    assert temp.cwt_coefs.shape[0] == temp.cwt_frequencies.shape[0]
    assert temp.cwt_coefs.shape == (15,200) 

def test_peak_fitting(): 
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    
    temp.get_peak_fitting()
    assert hasattr(temp, 'peak_features')
    assert temp.peak_features.shape == (60,)
    
    temp.get_peak_fitting(num_peaks=10)
    assert hasattr(temp, 'peak_features')
    assert temp.peak_features.shape == (30,)

def test_fft(): 
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    temp.pointwise_features(np.linspace(x.min(), x.max(),100))
    temp.fft_features()
    assert hasattr(temp, 'fft_feat')
    assert hasattr(temp, 'fft_freq')
    
    assert temp.fft_freq.shape == temp.fft_feat.shape == (100//2,)
    
    temp.pointwise_features(np.linspace(x.min(), x.max(),200))
    temp.fft_features()
    assert hasattr(temp, 'fft_feat')
    assert hasattr(temp, 'fft_freq')
    assert temp.fft_freq.shape == temp.fft_feat.shape == (200//2,)
    
    x, y = np.load('test_data/test_xas.npy')
    temp = spectral_features(x, y)
    temp.fft_features(xrange = np.linspace(x.min(), x.max(), 300))
    assert hasattr(temp, 'fft_feat')
    assert hasattr(temp, 'fft_freq')
    assert temp.fft_freq.shape == temp.fft_feat.shape == (300//2,)    

    
    