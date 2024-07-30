import sys
import os 
import pathlib 
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from spectrascope.features import *

x1, y1 = np.load('test_files/test_xas.npy')
fnX = 'test_files/X_testfile.npy'
fny = 'test_files/y_testfile.npy'
X = np.load(fnX)
X = X[0]
# y = np.load(fny)
x2 = np.linspace(4969.0, 5021.0, 200)
    
def test_spectral_features_class(): 
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x, y)
        
        assert len(temp.energies) == temp.intensities.shape[-1]
        assert type(temp.energies) == type(temp.intensities) == np.ndarray
    
def test_pointwise(): 
    x = np.linspace(4970, 5022, 100)
    y = np.random.randint(200, size=(100,))
    temp = spectral_features(x, y)
    
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x, y)
        
        xrange = np.linspace(4969.0, 5021.0, 200)
        assert len(temp.energies) == temp.intensities.shape[-1]
        
        temp.pointwise_features(xrange)
        
        assert hasattr(temp, 'xinterp')
        assert hasattr(temp, 'yinterp')
        
        assert len(temp.xinterp) == temp.yinterp.shape[-1] == 200
        
        xrange = np.linspace(4969.0, 5021.0, 300)
        temp.pointwise_features(xrange)
        assert len(temp.xinterp) == temp.yinterp.shape[-1] == 300
        
        xrange = np.linspace(4969.0, 5021.0, 100)
        temp.pointwise_features(xrange)
        assert len(temp.xinterp) == temp.yinterp.shape[-1] == 100
    
    

def test_polynomial():
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x,y)
        
        for N in [1,7,20]:
            for degree in [1, 3,5]: 
                temp.polynomial_featuresN(N, degree)
                assert hasattr(temp, f'polyfeatures_{N}_{degree}deg')
                featuresN = getattr(temp, f'polyfeatures_{N}_{degree}deg')
                assert featuresN.shape == (temp.intensities.shape[0], N * (degree+1)) 
                
        temp.polynomial_featuresN(7, 3, x=temp.energies[10:60], y=temp.intensities[:,10:60])
        assert hasattr(temp, f'polyfeatures_{round(temp.energies[10])}_{round(temp.energies[59])}_{7}_{3}deg')

def test_bins(): 
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x, y)
        
        for N in [1, 5,20]: 
            temp.generate_bins(N)
            assert hasattr(temp, f'bin_indices_{N}')
            bins_generated = getattr(temp, f'bin_indices_{N}')
            assert np.max(bins_generated) == N - 1 
            assert np.min(bins_generated) == 0 
            
            assert (np.unique(bins_generated) == np.arange(N)).all()
            
        temp.generate_bins(5, x = x[:55], y = temp.intensities[:,:55])
        minx = np.min(x)
        maxx = x[54]
        assert hasattr(temp, f'bin_indices_{round(minx)}_{round(maxx)}_{5}')
    

def test_polynomial_features_all(): 
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x, y)
        temp.polynomial_featuresall()
        assert hasattr(temp, 'polyfeatures_all')
        featuresall = getattr(temp, 'polyfeatures_all')
        assert featuresall.shape == (temp.intensities.shape[0],157) 
        
        temp.polynomial_featuresall(Narr = [3,1,4])
        assert hasattr(temp, 'polyfeatures_all')
        featuresall = getattr(temp, 'polyfeatures_all')
        assert featuresall.shape == (temp.intensities.shape[0], 3 * 4 + 1 * 4 + 4 * 4 + 1) 

        temp.polynomial_featuresall(degree=4, Narr = [3,10,7])
        assert hasattr(temp, 'polyfeatures_all')
        featuresall = getattr(temp, 'polyfeatures_all')
        assert featuresall.shape == (temp.intensities.shape[0], 3 * 5 + 10 * 5 + 7 * 5 + 1)     
    
    
def test_uneven():
    "print: uneven currently unsure"
    return  
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
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x, y)
        temp.cdf()
        assert hasattr(temp, 'cdf')
        assert temp.cdf_features.shape == temp.intensities.shape

def test_cwt(): 
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x, y)
        temp.cwt()
        assert hasattr(temp, 'cwt_coefs')
        assert hasattr(temp, 'cwt_frequencies')
        
        assert temp.cwt_coefs.shape[1] == temp.cwt_frequencies.shape[0]
        assert temp.cwt_coefs.shape[1:] == (10,len(temp.energies))
        
        temp.cwt(scales=np.arange(1,16))
        assert temp.cwt_coefs.shape[1] == temp.cwt_frequencies.shape[0]
        assert temp.cwt_coefs.shape[1:] == (15,len(temp.energies)) 
        
        temp.pointwise_features(xrange=np.linspace(x.min(), x.max(), 200))
        temp.cwt(scales=np.arange(1,16))
        assert temp.cwt_coefs.shape[1] == temp.cwt_frequencies.shape[0]
        assert temp.cwt_coefs.shape[1:] == (15,len(temp.xinterp)) 

def test_peak_fitting(): 
    for x, y in zip([x1], [y1]):  
        temp = spectral_features(x, y)
        
        temp.gaussian_peaks()
        assert hasattr(temp, 'peak_features')
        assert temp.peak_features.shape == (60,)
        
        temp.gaussian_peaks(num_peaks=10)
        assert hasattr(temp, 'peak_features')
        assert temp.peak_features.shape == (30,)

def test_fft(): 
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x, y)
        temp.pointwise_features(np.linspace(4969.0, 5021.0, 200))
        temp.fft_features()
        assert hasattr(temp, 'fft_feat')
        assert hasattr(temp, 'fft_freq')
        
        assert temp.fft_freq.shape[0] == temp.fft_feat.shape[-1] == (200 // 2)
        
        temp.pointwise_features(np.linspace(4969.0, 5021.0, 400))
        temp.fft_features()
        assert hasattr(temp, 'fft_feat')
        assert hasattr(temp, 'fft_freq')
        assert temp.fft_freq.shape[0] == temp.fft_feat.shape[-1] == 400//2
        
        # x, y = np.load('test_data/test_xas.npy')
        # temp = spectral_features(x, y)
        # temp.fft_features(xrange = np.linspace(x.min(), x.max(), 300))
        # assert hasattr(temp, 'fft_feat')
        # assert hasattr(temp, 'fft_freq')
        # assert temp.fft_freq.shape[0] == temp.fft_feat.shape[-1] == 300//2    

def test_gaussian_filter(): 
    for x, y in zip([x1,x2], [y1,X]):  
        temp = spectral_features(x,y)
        temp.pointwise_features(np.linspace(x.min(), x.max()),100)
        
        temp.gaussian_filter_features(sigma=1)
        assert hasattr(temp, 'gaussian_filter') 
        assert len(temp.gaussian_filter) == len(temp.yinterp)
        
        temp.pointwise_features(np.linspace(x.min(), x.max()),200)
        temp.gaussian_filter_features(sigma=5)
        assert hasattr(temp, 'gaussian_filter') 
        assert len(temp.gaussian_filter) == len(temp.yinterp)
        
        temp.gaussian_filter_features(sigma=0.5)
        assert hasattr(temp, 'gaussian_filter') 
        assert len(temp.gaussian_filter) == len(temp.yinterp)
    
