import sys
import os 
import pathlib 
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from spectrascope.data_processing import *

test_json = 'test_data/processed_billinge_Ti_test.json'
test_npy = ''
test_df = ''
def test_data_file_class():
     temp = data_file(test_json, 'Ti')
     
     assert len(temp.x_pdf) == len(temp.x_xanes)
     assert type(temp.best_fits) == dict 
     assert hasattr(temp, 'js')
     assert hasattr(temp, 'df')
     
     temp = data_file(test_json, 'Ti', random_seed=10)
     assert temp.random_seed == 10
     
     assert hasattr(temp, 'BL_mean')
    
def test_add_feature(): 
    temp = data_file(test_json, 'Ti')
    temp.add_feature_json(xarr=temp.x_xanes, feat='cdf', convcol='xanes')
    temp.add_feature_json(xarr=temp.x_pdf, feat='pointwise_features',convcol = 'pdf', xrange=temp.x_pdf)
    
    temp.add_feature_json(xarr=temp.x_xanes, feat='pointwise_features',convcol = 'xanes', xrange=temp.x_xanes)
    
    assert hasattr(temp, 'xanes_cdf')
    assert hasattr(temp, 'xanes_pointwise_features')
    assert hasattr(temp, 'pdf_pointwise_features')
    
    assert temp.xanes_pointwise_features.shape == temp.pdf_pointwise_features.shape
    assert np.unique(temp.xanes_cdf[:,-1])[0] == 1.0
    assert np.max(temp.xanes_cdf[0]) == 1.0
    
# def test_add_poly_feature(): 
#     temp = data_file(test_json, 'Ti')
#     temp.add_feature_json(temp.x_xanes, 'polynomial_featuresall', convcol='xanes', degree=3, Narr=np.array([4,5,10,20]), verbose=False)
#     assert hasattr(temp, 'xanes_polynomial_featuresall')
    
#     temp.add_feature_json(temp.x_xanes, 'polynomial_featuresall', convcol='pdf', degree=3, Narr=np.array([4,5,10,20]), verbose=False)
#     assert hasattr(temp, 'pdf_polynomial_featuresall')
    
#     assert temp.xanes_polynomial_featuresall.shape[1] == temp.pdf_polynomial_featuresall.shape[1] == 157
    
#     temp.add_feature_json(temp.x_xanes, 'polynomial_featuresall', convcol='pdf', degree=5, Narr=np.array([4,5,10]), verbose=False)
#     assert hasattr(temp, 'pdf_polynomial_featuresall')
#     assert temp.pdf_polynomial_featuresall.shape[1] == 55 
    
#     assert len(temp.df) == len(temp.pdf_polynomial_featuresall) == len(temp.xanes_polynomial_featuresall)
    
def test_combine_feature(): 
    temp = data_file(test_json, 'Ti')
    try: 
        temp.combine_feature_json('xanes_pointwise_features', 'pdf_pointwise_features')
    except AssertionError: 
        pass 
    
    temp.add_feature_json(xarr=temp.x_pdf, feat='pointwise_features',convcol = 'pdf', xrange=temp.x_pdf)
    
    temp.add_feature_json(xarr=temp.x_xanes, feat='pointwise_features',convcol = 'xanes', xrange=temp.x_xanes)
    
    temp.combine_feature_json('xanes_pointwise_features', 'pdf_pointwise_features')
    
    assert temp.xanes_pdf_pointwise_features.shape == (len(temp.df), len(temp.x_xanes)*2)
    
def test_split_X(): 
    temp = data_file(test_json, 'Ti')
    temp.add_feature_json(xarr=temp.x_xanes, feat='pointwise_features',convcol = 'xanes', xrange=temp.x_xanes)

    for test_size in [0.25, 0.5, 0.75]: 
        temp.split_X('xanes_pointwise_features', test_size)
        N, M  = temp.xanes_pointwise_features.shape
        Ntrain = np.floor(N * (1-test_size))
        Ntest = N - Ntrain
        assert temp.xanes_pointwise_features_train.shape == (Ntrain, 200)
        assert temp.xanes_pointwise_features_test.shape == (Ntest, 200)
        
    
    