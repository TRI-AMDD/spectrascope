import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import json 
from spectrascope.features import * 

import SPA 
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
import os 
import pickle


class SpectraScope:
    
    def __init__(self, X=None, y=None, fnX = None, fny = None, specnames=[], yname='', random_seed=None, save_path=None) -> None:
        
        date = datetime.date
        self.today = date.today().strftime('%m_%d_%Y')
        
        # add X 
        if X is not None:  
            self.X = X 
        elif fnX is not None: 
            self.X = self.load_file(fnX)

        
        if len(X) >= 2: 
            multi_modal = True
            self.n_modalities = len(X)
        else: 
            self.n_modalities = 0 
        
        assert len(specnames) == self.n_modalities, 'please input a name for each modality'
        
        for i, specname in enumerate(specnames): 
            Xi = self.X[i]
            setattr(self, specname, Xi)    
        
        # add Y 
        if y is not None: 
            setattr(self, yname, y) 
        elif fny is not None:
            y = self.load_file(fny)
            setattr(self, yname, y)  
        

        self.random_seed = random_seed
        if type(self.random_seed) is not int: 
            print('no random seed selected, when splitting data, will always give different split')
            
        self.best_fits = {}
        
        if save_path is None: 
            print('saving to /examples/models, change save_path kwarg if youd like to change')
            self.save_path = '/examples/models/'
        
        #### Need to incorporate this in some way    
        
    def load_file(self, fn):
        """
        Used by SS to load data files.
        """
        _, ext = os.path.splitext(fn)
        if ext == '.txt':
            for separator in (' ', ',', '\t', ';'): # Testing random separators
                data = pd.read_csv(fn, header = None, sep = separator).values
                if data.shape[-1] > 1: # We likely found the separator
                    break
        elif ext == '.csv':
            data = pd.read_csv(fn, header = None, sep = ',').values
        elif ext == '.tsv':
            data = pd.read_csv(fn, header = None, sep = '\t').values
        elif ext in {'.xls', '.xlsx'}:
            data = pd.read_excel(fn, header = None).values
        elif ext == '.npy':
            data = np.load(fn)
        elif ext == '.json': 
            data = pd.read_json(fn)
        else:
            raise ValueError(f'Please provide a filename with extension in {{.txt, .csv, .tsv, .xls, .xlsx, .npy, .json}}. You passed {filename}')
        return data
    
    def add_x_axis(self, specname, value):
        setattr(self, specname + '_xaxis', value) 
        
    def add_x_label(self, specname, value):
        setattr(self, specname + '_labelx', value) 
    
    def add_y_label(self, specname, value):
        setattr(self, specname + '_labely', value) 
                
    def add_feature(self, specname, feat, xaxis=None, additional_naming='', **kwargs):
        if xaxis is None: 
            xaxis = getattr(self, specname + '_xaxis')
        
        X = getattr(self, specname)
        new_feat = np.array([spectral_features(xaxis, X).__getattribute__(feat).__call__(return_features=True, **kwargs)])
        
        setattr(self, specname + '_' + feat + additional_naming, new_feat[0])
        
    # def add_feature_df(self, xarr, feat, convcol, additional_naming='', **kwargs): 
    #     self.df[f'{convcol}_' + feat + f'{additional_naming}'] = self.df[convcol].apply(lambda x: (spectral_features(xarr, x).__getattribute__(feat).__call__(return_features=True, **kwargs)))
        
    # def add_feature_json(self, xarr, feat, convcol, additional_naming='', add_to_json = False, **kwargs): 
    #     np_convcol = np.array([np.array(self.js[convcol][key]) for key in self.js[convcol]])  
        
        
    #     new_feat = np.array([spectral_features(xarr, x).__getattribute__(feat).__call__(return_features=True, **kwargs) for x in np_convcol])
        
        
    #     name = convcol + '_' + feat + additional_naming
    #     setattr(self, name, new_feat) 
        
    #     if add_to_json: 
    #         self.js[name] = new_feat
            
    # def add_y_json(self, feat, additional_naming=''): 
    #     setattr(self, feat + additional_naming, np.array([self.js[feat][key] for key in self.js[feat]]).astype(float))
        
    def combine_feature(self, feat1, feat2): 
        assert hasattr(self, feat1), 'please add feature first'
        assert hasattr(self, feat2), 'please add feature first'
        
        combined_feat = np.append(getattr(self, feat1), getattr(self, feat2), axis=1)
        setattr(self, feat1.split('_')[0] + '_' + feat2, combined_feat)
         
    def split_X(self, attr, test_size): 
        X = getattr(self, attr)
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=self.random_seed)
        
        setattr(self, attr + '_train', X_train)
        setattr(self, attr + '_test', X_test)
    
    def split_y(self, attr, test_size): 
        y = getattr(self, attr)
        y_train, y_test = train_test_split(y, test_size=test_size, random_state=self.random_seed)
        
        setattr(self, attr + '_train', y_train)
        setattr(self, attr + '_test', y_test)
    
    def sk_pipeline(self, X_train, X_test, y_train, y_test, model=RandomForestRegressor(), 
                scoring='neg_mean_squared_error',cv=5,
                param_grid: dict = {'n_estimators':[50,100], 'criterion':['squared_error']},
                refit=True, metric = mean_squared_error, savef=None):
    
        P = make_pipeline(GridSearchCV(model, 
                                        param_grid=param_grid, 
                                        cv=cv, 
                                        refit=refit, 
                                        scoring=scoring,
                                        n_jobs=-1
                                        ))
        P.fit(X_train, y_train)
        ypred = P.predict(X_test)
        score = metric(y_test, ypred)
        
        # if savef is not None: 
        #     with open(savef, 'wb') as f: 
        #         pickle.dump(P,f)
        
        return P, score, ypred
    
        
    def do_sk(self, Xattr, yattr, test_size, savef=None, **kwargs): 
        
        # if not hasattr(self, Xattr + '_train'):
        self.split_X(Xattr, test_size)
        # if not hasattr(self, yattr + '_test'):
        self.split_y(yattr, test_size)
        
        X_train = getattr(self, Xattr + '_train')
        X_test = getattr(self, Xattr + '_test')
        
        y_train = getattr(self, yattr + '_train')
        y_test = getattr(self, yattr + '_test')
        
        P, score, ypred = self.sk_pipeline(X_train, X_test, y_train, y_test, **kwargs)
        model_name = P[0].best_estimator_.__str__().split('(')[0]
        fit_dict = {'P':P, 'score':score, 'ypred':ypred}
        setattr(self, Xattr + '_' + yattr + '_' + model_name, fit_dict)
        
        perc_err = np.mean(np.abs((y_test - ypred) / y_test)) * 100
        
        if model_name not in self.best_fits: 
            self.best_fits[model_name] = {Xattr + '_' + yattr : perc_err} #score
        else: 
            self.best_fits[model_name][Xattr + '_' + yattr] = perc_err  #score
        
        if savef is not None: 
            self.check_dir_exists(f'{self.save_path}/{self.today}')
            with open(f'{self.save_path}/{self.today}/sk_{model_name}_{self.timenow}_{Xattr}_{yattr}_{savef}.p', 'wb') as f: 
                pickle.dump(P,f)
        
    def do_LCEN(self, Xattr, yattr, test_size, savef=None, **kwargs):
        
        # if not hasattr(self, Xattr + '_train'):
        self.split_X(Xattr, test_size)
        # if not hasattr(self, Xattr + '_test'):
        self.split_y(yattr, test_size)
        
        X_train = getattr(self, Xattr + '_train')
        X_test = getattr(self, Xattr + '_test')
        
        y_train = getattr(self, yattr + '_train')
        y_test = getattr(self, yattr + '_test')
        
        #TODO: have to save filename for LCEN; update it so I don't? 
        np.save('X_train_temp', X_train)
        np.save('X_test_temp', X_test)
        np.save('y_train_temp', y_train)
        np.save('y_test_temp', y_test)
        
        out = SPA.main_SPA('X_train_temp.npy', 'y_train_temp.npy', 'X_test_temp.npy', 'y_test_temp.npy', model_name=['LCEN'], **kwargs)
        
        setattr(self, Xattr + '_' + yattr + '_LCEN', out[0]['LCEN'])
        score = out[0]['LCEN']['MSE_test_nontrans']
        ypred = out[0]['LCEN']['yhat_test_nontrans']
        perc_err = np.mean(np.abs((y_test - ypred) / y_test)) * 100
        
        if 'LCEN' not in self.best_fits: 
            self.best_fits['LCEN'] = {Xattr + '_' + yattr : perc_err} #score
        else: 
            self.best_fits['LCEN'][Xattr + '_' + yattr] = perc_err  #score
            
        if savef is not None: 
            self.check_dir_exists(f'{self.save_path}/{self.today}')
            with open(f'{self.save_path}/{self.today}/LCEN_{self.timenow}_{Xattr}_{yattr}_{savef}.p', 'wb') as f: 
                pickle.dump(out[0]['LCEN'],f)
        
    def do_all(self, feats:dict, yattr, test_size, sk_kwargs=[], lcen_kwargs=None, multi_modal=None, savef=None):
        """

        Args:
            feats (dict): {feature: [feature_kwargs1, feature_kwargs2]}
            y (attr name): name of yattr to regress on 
            test_size (float/int): input for sklearn train_test_split test_size 
            multi_modal (list, optional): [(feat1,feat2),]. Defaults to None. 
        """
        self.timenow = datetime.datetime.now().time().strftime('%H_%M_%S_%f')
        for feat in feats: 
            kwargs_list = feats[feat]
            for kwargs in kwargs_list: 
                attr_name = kwargs['specname'] + '_' + feat
                if not hasattr(self, attr_name): 
                    self.add_feature(feat=feat, **kwargs)
                            
                for sk in sk_kwargs: 
                    self.do_sk(attr_name, yattr, test_size, savef, **sk)
                if lcen_kwargs is not None:
                    self.do_LCEN(attr_name, yattr, test_size, savef, **lcen_kwargs)
            
        if multi_modal is not None: 
            for combo in multi_modal: 
                feat1, feat2 = combo
                attr_name = feat1.split('_')[0] + '_' + feat2
                if not hasattr(self, attr_name): 
                    self.combine_feature(feat1, feat2)
                
                for sk in sk_kwargs: 
                    self.do_sk(attr_name, yattr, test_size, savef, **sk)
                if lcen_kwargs is not None: 
                    self.do_LCEN(attr_name, yattr, test_size, savef, **lcen_kwargs)
            
    def check_dir_exists(self, directory): 
        try: 
            os.makedirs(directory)
        except OSError: 
            pass 
    
    
    # these might be unnecessary actually 
    # def add_df_poly_all(self, degree, convcol = 'spectrum', Narr=np.array([4,5,10,20]), additional_naming=''): 
    #     self.df['spectrum_poly_' + '_'.join([str(num) for num in Narr]) + f'{additional_naming}'] = self.df[convcol].apply(lambda x: spectral_features(x['x'], x['y']).polynomial_featuresall(Narr=Narr, degree=degree, return_features=True, verbose=False))
    #     # return df

    # def add_df_poly_individual(self, N, degree, convcol = 'spectrum', additional_naming=''):
    #     self.df[f'spectrum_poly_{N}' + f'{additional_naming}'] = self.df[convcol].apply(lambda x: spectral_features(x['x'], x['y']).polynomial_featuresN(N=N, degree=degree, return_features=True, verbose=False).ravel())
    #     # return df 

    # def add_df_pointwise(self, convcol = 'spectrum', custom_range = None, additional_naming=''):
    #     self.df['spectrum_point' + f'{additional_naming}'] = self.df[convcol].apply(lambda x: spectral_features(x['x'], x['y']).pointwise_features(custom_range=custom_range, return_features=True))
    #     # return df
        
    # def add_df_poly_all_billinge(self, degree, xarr, convcol = 'xanes', Narr=np.array([4,5,10,20]), additional_naming='', verbose=False): 
    #     self.df['spectrum_poly_' + '_'.join([str(num) for num in Narr]) + f'{additional_naming}'] = self.df[convcol].apply(lambda x: spectral_features(xarr, x).polynomial_featuresall(Narr=Narr, degree=degree, return_features=True, verbose=verbose))

    # def add_df_poly_individual_billinge(self, N, xarr, degree, convcol = 'xanes', additional_naming='', verbose=False): 
    #     self.df[f'spectrum_poly_{N}' + f'{additional_naming}'] = self.df[convcol].apply(lambda x: spectral_features(xarr, x).polynomial_featuresN(N=N, degree=degree, return_features=True, verbose=verbose).ravel())

         
        