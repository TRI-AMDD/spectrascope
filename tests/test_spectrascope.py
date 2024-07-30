import sys
import os 
import pathlib 
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from spectrascope.SpectraScope import *

test_json = ''
test_npy = ''
test_df = ''

fnX = 'test_files/X_testfile.npy'
fny = 'test_files/y_testfile.npy'
X = np.load(fnX)
y = np.load(fny)

def test_spectrascope_class():
     temp = SpectraScope(X = X, y = y, specnames=['pdf', 'xanes'], yname='nn')
     
    
     assert type(temp.best_fits) == dict 
     assert hasattr(temp, 'xanes')
     assert hasattr(temp, 'pdf')
     assert len(temp.xanes) == len(temp.pdf)
     
     temp = SpectraScope(X = X, y = y, specnames=['pdf', 'xanes'], yname='nn', random_seed=10)
     assert temp.random_seed == 10
     
     assert hasattr(temp, 'nn')
    
def test_add_feature(): 
    temp = SpectraScope(X = X, y = y, specnames=['pdf', 'xanes'], yname='nn')
    
    temp.add_x_axis(specname='xanes', value=np.linspace(4969.0, 5021.0, 200))
    temp.add_x_axis(specname='pdf', value=np.linspace(0,10,200))

    temp.add_feature(feat='cdf', specname='xanes')
    temp.add_feature(feat='pointwise_features',specname = 'pdf', xrange=temp.xanes_xaxis)
    
    temp.add_feature(feat='pointwise_features',specname = 'xanes', xrange=temp.pdf_xaxis)
    
    assert hasattr(temp, 'xanes_cdf')
    assert hasattr(temp, 'xanes_pointwise_features')
    assert hasattr(temp, 'pdf_pointwise_features')
    
    assert temp.xanes_pointwise_features.shape == temp.pdf_pointwise_features.shape
    assert np.unique(temp.xanes_cdf[:,-1])[0] == 1.0
    assert np.max(temp.xanes_cdf[0]) == 1.0
    
def test_combine_feature(): 
    temp = SpectraScope(X = X, y = y, specnames=['pdf', 'xanes'], yname='nn')
    
    temp.add_feature(xaxis = np.linspace(4969.0, 5021.0, 200), feat='cdf', specname='xanes')
    temp.add_feature(xaxis = np.linspace(0,10,200), feat='pointwise_features',specname = 'pdf', xrange=np.linspace(0,10,200))
    
    try: 
        temp.combine_feature('xanes_pointwise_features', 'pdf_pointwise_features')
    except AssertionError: 
        pass 
    
    temp.add_x_axis(specname='xanes', value=np.linspace(4969.0, 5021.0, 200))
    temp.add_x_axis(specname='pdf', value=np.linspace(0,10,200))
        
    temp.add_feature(feat='pointwise_features',specname = 'xanes', xrange=temp.xanes_xaxis)
    
    temp.combine_feature('xanes_pointwise_features', 'pdf_pointwise_features')
    
    assert temp.xanes_pdf_pointwise_features.shape == (len(temp.xanes), len(temp.xanes_xaxis)*2)
    
def test_split_X(): 
    temp = SpectraScope(X = X, y = y, specnames=['pdf', 'xanes'], yname='nn')
    temp.add_x_axis(specname='xanes', value=np.linspace(4969.0, 5021.0, 200))
    temp.add_feature(feat='pointwise_features',specname = 'xanes', xrange=temp.xanes_xaxis)

    for test_size in [0.25, 0.5, 0.75]: 
        temp.split_X('xanes_pointwise_features', test_size)
        N, M  = temp.xanes_pointwise_features.shape
        Ntrain = np.floor(N * (1-test_size))
        Ntest = N - Ntrain
        assert temp.xanes_pointwise_features_train.shape == (Ntrain, 200)
        assert temp.xanes_pointwise_features_test.shape == (Ntest, 200)

    