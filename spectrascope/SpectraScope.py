"""Core SpectraScope class for spectral data loading, feature generation, model training, and evaluation.

The main entry point is the :class:`SpectraScope` class, which manages one or more spectral
datasets, coordinates feature extraction via :mod:`spectrascope.Features`, and provides
wrappers for sklearn pipelines and the LCEN regression model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from spectrascope.Features import *

import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
import os
import pickle
import spectrascope.StructurePropertyFeatures as structure_property_featurization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from spectrascope.utils import LCEN_selected_features, maximal_correlation_from_ace
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.base import clone
from tqdm import tqdm

__docformat__ = "numpy"


class SpectraScope:
    """The main SpectraScope class encompasses all data loading, feature generation, model training, and model evaluation."""

    def __init__(
        self,
        X=None,
        y=None,
        fnX: str = None,
        fny: str = None,
        specnames: list = [],
        yname: str = "",
        random_seed: int = None,
        save_path: str = None,
    ) -> None:
        """
        Can load X, y via an array or file path. If file path, will try to infer file type.

        Data will be referred to as "spectrums" but can be any 1D array like data.

        Parameters
        ----------
        X : arr, optional
            Input X. Defaults to None.
        y : arr, optional
            Input y. Defaults to None.
        fnX : str, optional
            Input X file path. Defaults to None.
        fny : str, optional
            Input y file path. Defaults to None.
        specnames : list, optional
            Names of different spectra that have loaded, if loading multiple modalities. Defaults to [].
        yname : str, optional
            Name of yprediction. Defaults to "".
        random_seed : int, optional
            Random seed for splitting data and training. Suggested to set a random seed to keep same splits when comparing across models. Defaults to None.
        save_path : str, optional
            Path for saving results. Defaults to None (saves to /examples/models).

        Attributes
        ----------
        today : str
            Today's date in format MM_DD_YYYY.
        X : arr
            Input X.
        y : arr
            Input y.
        best_fits : dict
            Dictionary of best fits.
        specnames : list
            Names of different spectra that have loaded.
        random_seed : int
            Random seed for splitting data and training.
        save_path : str
            Path for saving results.
        """
        date = datetime.date
        self.today = date.today().strftime("%m_%d_%Y")

        # add X
        if X is not None:
            if isinstance(X, list):
                self.X = X
            else:
                self.X = [X]
            self.X = X
        elif fnX is not None:
            self.X = self.load_file(fnX)

        if specnames == []:
            specnames = [f"spec{i}" for i in range(len(self.X))]

        for i, specname in enumerate(specnames):
            Xi = self.X[i]
            setattr(self, specname, Xi)

        # add Y
        if y is not None:
            if yname == "":
                yname = "y0"
            self.add_y(yname, y)
        elif fny is not None:
            if yname == "":
                yname = "y0"
            y = self.load_file(fny)
            self.add_y(yname, y)

        if type(random_seed) is not int:
            print(
                "no random seed selected, when splitting data, will always give different split"
            )
            self.random_seed = random_seed
        else:
            self.random_seed = random_seed

        self.best_fits = {}
        self.specnames = specnames
        self.generated_features = []
        if save_path is None:
            print(
                "saving to /examples/models, change save_path kwarg if youd like to change"
            )
            self.save_path = "/examples/models/"

    def add_generated_feature(self, feat):
        """Register a feature name in the generated features list (idempotent).

        Parameters
        ----------
        feat : str
            Name of the feature attribute to register.
        """
        if feat not in self.generated_features:
            self.generated_features.append(feat)
        else:
            pass

    def load_file(self, fn: str):
        """Load data from a file.

        Parameters
        ----------
        fn : str
            The file path.

        Returns
        -------
        data : np.array
            The loaded data.
        """

        _, ext = os.path.splitext(fn)
        if ext == ".txt":
            for separator in (" ", ",", "\t", ";"):  # Testing random separators
                data = pd.read_csv(fn, header=None, sep=separator).values
                if data.shape[-1] > 1:  # We likely found the separator
                    break
        elif ext == ".csv":
            data = pd.read_csv(fn, header=None, sep=",").values
        elif ext == ".tsv":
            data = pd.read_csv(fn, header=None, sep="\t").values
        elif ext in {".xls", ".xlsx"}:
            data = pd.read_excel(fn, header=None).values
        elif ext == ".npy":
            data = np.load(fn)
        elif ext == ".json":
            data = pd.read_json(fn)
        else:
            raise ValueError(
                f"Please provide a filename with extension in {{.txt, .csv, .tsv, .xls, .xlsx, .npy, .json}}. You passed {filename}"
            )
        return data.to_numpy()

    def add_x_axis(self, specname: str, value: np.array):
        """
        Add x-axis attribute for a specific spectrum. Attribute name is {specname}_xaxis.

        Parameters
        ----------
        specname : str
            Spectrum to associate x-axis with.
        value : np.array
            x-axis values.
        """
        setattr(self, specname + "_xaxis", value)

    def add_x_label(self, specname: str, value: str):
        """Add x-axis label attribute for a specific spectrum. Attribute name is {specname}_labelx.

        Parameters
        ----------
        specname : str
            spectrum to associate x-axis label with.
        value : str
            x-axis label.
        """
        setattr(self, specname + "_labelx", value)

    def add_y_label(self, specname: str, value: str):
        """Add y-axis label attribute for a specific spectrum. Attribute name is {specname}_labely.

        Parameters
        ----------
        specname : str
            spectrum to associate y-axis label with.
        value : str
            y-axis label.
        """
        setattr(self, specname + "_labely", value)

    def scale_X(self, X: np.array) -> np.array:
        """Apply sklearn StandardScaler to the input data.

        Standardize features by removing the mean and scaling to unit variance.

        Parameters
        ----------
        X : np.array
            Input data to scale.

        Returns
        -------
        scaled_X : np.array
            Scaled X.
        """
        scaled_X = StandardScaler().fit_transform(X)
        return scaled_X

    def add_spectral_feature(
        self,
        specname: str,
        feat: str,
        xaxis=None,
        additional_naming: str = "",
        **kwargs,
    ):
        """Add a spectral feature attribute to the SpectraScope object for a specific spectrum.

        Attribute name is {specname}_{feat}{additional_naming}.

        Parameters
        ----------
        specname : str
            Name of the spectrum to featurize / transform.
        feat : str
            Name of the feature to add. This must correspond to a method in the spectral_features module.
        xaxis : np.array, optional
            x-axis values for the spectrum. Defaults to None.
        additional_naming : str, optional
            Additional naming for the default attribute. Defaults to "".
        **kwargs :
            Additional keyword arguments to pass to the spectral_features method.

        """
        if xaxis is None:
            xaxis = getattr(self, specname + "_xaxis")

        X = getattr(self, specname)
        new_feat = np.array(
            [
                SpectralFeatures(xaxis, X)
                .__getattribute__(feat)
                .__call__(return_features=True, **kwargs)
            ]
        )
        full_name = specname + "_" + feat + additional_naming
        setattr(self, full_name, new_feat[0])
        self.add_generated_feature(full_name)

    def add_y(self, yname: str, y: np.array):
        """Add a new prediction target y attribute to the SpectraScope object.

        Attribute name is {yname}.

        Parameters
        ----------
        yname : str
            Name of target variable.
        y : np.array
            Value of target variable. Must be = len(X).
        """
        setattr(self, yname, y)

    def add_structural_label(
        self, structarr: str, structfeat: str, additional_naming="", **kwargs
    ):
        """Not Implemented yet.

        Parameters
        ----------
        structarr : str
            Name of structure array attribute
        structfeat : _type_
            Name of the structure_property_featurization method to call
        additional_naming : str, optional
            _description_, by default ""
        **kwargs : _type_
            Additional kwargs for the structure_property_featurization method called
        """
        structure_arr = getattr(self, structarr)
        struct_label = getattr(structure_property_featurization, structfeat)

        y = []
        for struct in structure_arr:
            y.append(int(struct_label.__call__(structure=struct, **kwargs)))
        setattr(
            self,
            structarr
            + "_"
            + structfeat
            + "_"
            + "_".join([str(val) for val in kwargs.values()])
            + additional_naming,
            y,
        )
        # self.structural_features()

    def structural_labels(self):
        """Not implemented yet.

        Collect all structural labels into a single array.
        """
        y_structural = np.array(
            [
                getattr(self, struct_attr)
                for struct_attr in dir(self)
                if "structures_" in struct_attr
            ]
        )
        setattr(self, "y_structural", y_structural)
        names = [
            struct_attr for struct_attr in dir(self) if "structures_" in struct_attr
        ]
        setattr(self, "y_structural_names", np.array(names))

    def combine_feature(self, feat1: str, feat2: str, additional_naming=""):
        """Concatenate two features and save as a new attribute for multimodal training.

        Features must exist before combining. Naming assumes that the two features are the same transformation on different types of experimental data. Attribute name is {spec1}_{feat2}{additional_naming}.

        Parameters
        ----------
        feat1 : str
            Name for first feature to combine.
        feat2 : str
            Name for second feature to combine.
        additional_naming : str, optional
            Additional naming for new attribute. Default "".
        """
        assert hasattr(self, feat1), "please add feature first"
        assert hasattr(self, feat2), "please add feature first"

        combined_feat = np.append(getattr(self, feat1), getattr(self, feat2), axis=1)
        full_name = feat1.split("_")[0] + "_" + feat2 + additional_naming
        setattr(self, full_name, combined_feat)
        self.add_generated_feature(full_name)

    def transform_feature(
        self,
        feat: str,
        degree: int = 2,
        interaction: bool = True,
        trans_type: str = "all",
        all_pos_y: bool = True,
        additional_naming: str = "",
    ):
        """Adds new attribute of polynomial etc. transformations of already generated features.

        Uses _feature_trans function from regression_models in SPA v1.4.1. This version is key to avoid errors.
        Attribute of array is {feat}_trans{additional_naming}.
        Attribute of names of transformations is {feat}_trans_names{additional_naming}.

        Parameters
        ----------
        feat : str
            Feature attribute to transform. Will transform test if exists. e.g. "spec1_pointwise_features"
        degree : int, optional
            Degree of polynomial transformations. Must be in [1,2,3]. Default 2
        interaction : bool, optional
            _description_, by default True
        trans_type : str in {'all', 'poly', 'simple_interaction'}, optional, default "all"
            Whether to include all transforms (polynomial, log, sqrt, inverse) only polynomial transform, or just interaction terms.
        all_pos_y : bool, optional, default=True
             Whether all entries in the data are positive or also contain some negative entries. This is done to avoid taking sqrt(k) for some k<0.
        additional_naming : str, optional, default ""
            Additional naming for final feature.

        See Also
        --------
        regression_models._feature_trans in SPA module for more details.
        """

        import regression_models

        X = getattr(self, feat)
        try:
            X_test = getattr(self, feat + "_test")
        except AttributeError:
            X_test = None
        all_pos_X = np.all(X >= 0, axis=0)
        X_out, X_test_out, names = regression_models._feature_trans(
            X=X,
            X_test=X_test,
            degree=degree,
            interaction=interaction,
            trans_type=trans_type,
            all_pos_X=all_pos_X,
            all_pos_y=all_pos_y,
        )

        setattr(self, feat + "_trans" + additional_naming, X_out)
        if X_test_out is not None:
            setattr(self, feat + "_test_trans" + additional_naming, X_test_out)
        setattr(self, feat + "_trans_names" + additional_naming, names)

        self.add_generated_feature(feat + "_trans" + additional_naming)

    def split_X(self, attr: str, test_size, scale_X=False):
        """Split X attribute using sklearn train_test_split.

        Parameters
        ----------
        attr : str
            Name of the X attribute to split.
        test_size : float or int
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
        scale_X : bool, optional, default=False
            Whether or not to scale the X data first.
        """
        X = getattr(self, attr)
        if scale_X:
            X = self.scale_X(X)

        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=self.random_seed
        )

        setattr(self, attr + "_train", X_train)
        setattr(self, attr + "_test", X_test)

    def split_y(self, attr, test_size):
        """Split a target variable into train and test sets.

        Sets ``{attr}_train`` and ``{attr}_test`` attributes on self.

        Parameters
        ----------
        attr : str
            Name of the y attribute to split.
        test_size : float or int
            Fraction (float) or absolute count (int) for the test set; passed to
            ``sklearn.model_selection.train_test_split``.
        """
        y = getattr(self, attr)
        y_train, y_test = train_test_split(
            y, test_size=test_size, random_state=self.random_seed
        )

        setattr(self, attr + "_train", y_train)
        setattr(self, attr + "_test", y_test)

    def sk_pipeline(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        estimator=RandomForestRegressor(),
        scoring="r2",
        cv=5,
        param_grid: dict = {"n_estimators": [50, 100], "criterion": ["squared_error"]},
        refit=True,
        metric=mean_squared_error,
        savef=None,
    ):
        """Build pipeline for sklearn model training and hyperparameter tuning using GridSearchCV.

        Parameters
        ----------
        X_train : np.array
            train data.
        X_test : np.array
            test data.
        y_train : np.array
            train labels.
        y_test : np.array
            test labels.
        estimator : _type_, optional, default RandomForestRegressor()
            initialized sklearn estimator for training
        scoring : str, optional, default "r2"
            scoring metric for GridSearchCV (e.g. "r2", "neg_mean_squared_error")
        cv : int, optional, default 5
            folds for k-fold cross validation
        param_grid : dict, optional
            param grid for GridSearchCV, by default {"n_estimators": [50, 100], "criterion": ["squared_error"]}
        refit : bool, optional, default True
            whether to refit an estimator using the best found parameters on the whole dataset. best_estimator is this final estimator.
        metric : function, optional
            sklearn metric, or any metric function for score, by default mean_squared_error
        savef : str or None, optional, default None
            path to save final model; not implemented currently.

        Returns
        -------
        P : sklearn.pipeline.Pipeline
            fitted pipelen
        score : float
            metric score of best estimator
        ypred : np.array
            predicted y values
        """

        try:
            check_is_fitted(estimator)
            print("already fitted")
            estimator = clone(estimator)
        except NotFittedError as exc:
            # print(f"Model is not fitted yet.")
            pass

        P = make_pipeline(
            GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                refit=refit,
                scoring=scoring,
                n_jobs=-1,
            )
        )
        P.fit(X_train, y_train)
        ypred = P.predict(X_test)
        score = metric(y_test, ypred)

        # if savef is not None:
        #     with open(savef, 'wb') as f:
        #         pickle.dump(P,f)

        return P, score, ypred

    def sk(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        estimator=RandomForestRegressor(),
        metric=mean_squared_error,
        savef=None,
        **kwargs,
    ):
        """Fit sklearn estimator without hyperparameter tuning.

        Parameters
        ----------
        X_train : np.array
            train data.
        X_test : np.array
            test data.
        y_train : np.array
            train labels.
        y_test : np.array
            test labels.
        estimator : _type_, optional, default RandomForestRegressor()
            initialized sklearn estimator for training.
        metric : function, optional
            sklearn metric, or any metric function for score, by default mean_squared_error
        savef : str or None, optional, default None
            path to save final model; not implemented currently.

        Returns
        -------
        est : _type_
            fitted estimator.
        score : float
            metric score of fitted estimator.
        ypred : np.array
            predicted y values
        """
        try:
            check_is_fitted(estimator)
            print("already fitted")
            estimator = clone(estimator)
        except NotFittedError as exc:
            # print(f"Model is not fitted yet.")
            pass

        est = estimator
        est.fit(X_train, y_train)
        ypred = est.predict(X_test)
        score = metric(y_test, ypred)
        return est, score, ypred

    def do_sk(
        self,
        Xattr: str,
        yattr: str,
        test_size=0.25,
        savef=None,
        use_pipeline=True,
        top_features=20,
        scale_X=True,
        **kwargs,
    ):
        """Fit sklearn model or pipeline and save results.

        Parameters
        ----------
        Xattr : str
            X attribute to fit model on.
        yattr : str
            y attribute to fit model on.
        test_size : float or int, optional, default 0.25
            input for sklearn train_test_split test_size.
        savef : str or None, optional, default None
            filename to save model.
        use_pipeline : bool, optional, default=True
            whether to use sklearn pipeline and GridSearchCV.
        top_features : int, optional, default=20.
            number of top features to save for future analysis.
        scale_X : bool, optional, default=True.
            whether or not to scale X data before fitting using StandardScaler.
        """
        self.split_X(Xattr, test_size, scale_X=scale_X)
        self.split_y(yattr, test_size)

        X_train = getattr(self, Xattr + "_train")
        X_test = getattr(self, Xattr + "_test")

        y_train = getattr(self, yattr + "_train")
        y_test = getattr(self, yattr + "_test")

        if use_pipeline:
            P, score, ypred = self.sk_pipeline(
                X_train, X_test, y_train, y_test, **kwargs
            )
            model_name = P[0].best_estimator_.__str__().split("(")[0]
        else:
            P, score, ypred = self.sk(X_train, X_test, y_train, y_test, **kwargs)
            model_name = P.__str__().split("(")[0]

        if "RandomForest" in model_name:
            try:
                feature_importances = P.feature_importances_
            except AttributeError:
                feature_importances = P[0].best_estimator_.feature_importances_
            # choose top features
            selected_features = np.argsort(feature_importances[::-1])[:top_features]
            selected_vals = feature_importances[np.argsort(feature_importances[::-1])][
                :top_features
            ]
        elif "LogisticRegression" in model_name:
            print(top_features)
            selected_features = np.argsort(np.abs(P.coef_))[::-1][:top_features]
            selected_vals = P.coef_[np.argsort(np.abs(P.coef_))[::-1]][:top_features]
        elif "FusedLasso" in model_name:
            selected_features = []
            selected_vals = []
        else:
            selected_features = []
            selected_vals = []
            # changes dataset: (X[:,sorted(sorted_best_features)])

        fit_dict = {
            "P": P,
            "score": score,
            "ypred": ypred,
            "selected_features": selected_features,
            "scaled": scale_X,
            "selected_vals": selected_vals,
        }
        setattr(self, Xattr + "_" + yattr + "_" + model_name, fit_dict)

        perc_err = np.mean(np.abs((y_test - ypred) / y_test)) * 100

        if model_name not in self.best_fits:
            self.best_fits[model_name] = {Xattr + "_" + yattr: score}  # score
        else:
            self.best_fits[model_name][Xattr + "_" + yattr] = score  # score

        if savef is not None:
            self.check_dir_exists(f"{self.save_path}/{self.today}")
            with open(
                f"{self.save_path}/{self.today}/sk_{model_name}_{self.timenow}_{Xattr}_{yattr}_{savef}.p",
                "wb",
            ) as f:
                pickle.dump(P, f)

    def do_LCEN(
        self,
        Xattr: str,
        yattr: str,
        test_size=0.25,
        savef=None,
        top_features=20,
        **kwargs,
    ):
        """Fit LCEN model using SPA and save results.

        Parameters
        ----------
        Xattr : str
            X attribute to fit model on.
        yattr : str
            y attribute to fit model on.
        test_size : float or int, optional, default 0.25
            input for sklearn train_test_split test_size.
        savef : str or None, optional, default=None
            filename to save model, default None. Currently SPA automatically saves model.
        top_features : int, optional, default=20
            number of top features to save for future analysis.
        """
        import SPA

        self.split_X(Xattr, test_size)
        self.split_y(yattr, test_size)

        X_train = getattr(self, Xattr + "_train")
        X_test = getattr(self, Xattr + "_test")

        y_train = getattr(self, yattr + "_train")
        y_test = getattr(self, yattr + "_test")

        if "metric" not in kwargs:
            metric = "MAE"
        else:
            metric = kwargs["metric"]
            kwargs.pop("metric")

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as _tmpdir:
            _orig_dir = os.getcwd()
            try:
                os.chdir(_tmpdir)
                np.save("X_train_temp", X_train)
                np.save("X_test_temp", X_test)
                np.save("y_train_temp", y_train)
                np.save("y_test_temp", y_test)
                out = SPA.main_SPA(
                    "X_train_temp.npy",
                    "y_train_temp.npy",
                    "X_test_temp.npy",
                    "y_test_temp.npy",
                    model_name=["LCEN"],
                    **kwargs,
                )
            finally:
                os.chdir(_orig_dir)
        vals, selected_features = LCEN_selected_features(out[0]["LCEN"], top_features)
        out[0]["LCEN"]["scaled"] = True
        out[0]["LCEN"]["selected_features"] = selected_features
        out[0]["LCEN"]["selected_vals"] = vals

        setattr(self, Xattr + "_" + yattr + "_LCEN", out[0]["LCEN"])
        ypred = out[0]["LCEN"]["yhat_test_nontrans"]
        perc_err = np.mean(np.abs((y_test - ypred) / y_test)) * 100

        if metric == "MAE":
            score = out[0]["LCEN"]["MSE_test_nontrans"]
        else:
            score = metric(y_test, ypred)

        if "LCEN" not in self.best_fits:
            self.best_fits["LCEN"] = {Xattr + "_" + yattr: score}  # score
        else:
            self.best_fits["LCEN"][Xattr + "_" + yattr] = score  # score
        out[0]["LCEN"]["score"] = score
        if savef is not None:
            self.check_dir_exists(f"{self.save_path}/{self.today}")
            with open(
                f"{self.save_path}/{self.today}/LCEN_{self.timenow}_{Xattr}_{yattr}_{savef}.p",
                "wb",
            ) as f:
                pickle.dump(out[0]["LCEN"], f)

    def do_all(
        self,
        feats: dict,
        yattr,
        test_size,
        sk_kwargs=[],
        lcen_kwargs=None,
        multi_modal=None,
        savef=None,
        do_all_existing=False,
        use_pipeline=True,
        scale_X=True,
    ):
        """Perform a list of transformations and fitting.

        Currently, add transformations separately because transformations aren't part of add feature function.

        Parameters
        ----------
        feats : dict
            {feature: [feature_kwargs1, feature_kwargs2]}
        yattr : str
            name of yattr to regress on
        test_size : float or int
            input for sklearn train_test_split test_size
        sk_kwargs : list, optional, default=[]
            list of additional kwargs for sklearn models.
        lcen_kwargs: dict, optional, default=None
            kwargs for LCEN model.
        multi_modal : list, optional, default=None
            [(feat1,feat2),]. Defaults to None.
        savef : str, optional, default=None
            filename to save model.
        do_all_existing : bool, optional, default=False
            If True, will run on all existing features. If False, will run only features in feats.
        use_pipeline : bool, optional, default=False
            If True, will use sklearn pipeline and GridSearchCV.
        scale_X : bool, optional, default=True
            If True, will scale X data before fitting.
        """
        self.timenow = datetime.datetime.now().time().strftime("%H_%M_%S_%f")

        if do_all_existing:
            for spec in self.specnames:
                attr_names = [
                    feat
                    for feat in dir(self)
                    if "feature" in feat
                    and f"{spec}_" in feat
                    and type(getattr(self, feat)) == np.ndarray
                    and "names" not in feat
                    and "test" not in feat
                    and "train" not in feat
                ]
                for attr_name in attr_names:
                    for sk in sk_kwargs:
                        self.do_sk(
                            attr_name, yattr, test_size, savef, scale_X=scale_X, **sk
                        )
                    if lcen_kwargs is not None:
                        self.do_LCEN(attr_name, yattr, test_size, savef, **lcen_kwargs)

        else:

            for feat in feats:
                kwargs_list = feats[feat]
                for kwargs in kwargs_list:
                    attr_name = kwargs["specname"] + "_" + feat
                    if not hasattr(self, attr_name):
                        self.add_spectral_feature(feat=feat, **kwargs)

                    for sk in sk_kwargs:
                        self.do_sk(
                            attr_name,
                            yattr,
                            test_size,
                            savef,
                            use_pipeline=use_pipeline,
                            scale_X=scale_X,
                            **sk,
                        )
                    if lcen_kwargs is not None:
                        self.do_LCEN(attr_name, yattr, test_size, savef, **lcen_kwargs)

            if multi_modal is not None:
                for combo in multi_modal:
                    feat1, feat2 = combo
                    attr_name = feat1.split("_")[0] + "_" + feat2
                    if not hasattr(self, attr_name):
                        self.combine_feature(feat1, feat2)

                    for sk in sk_kwargs:
                        self.do_sk(
                            attr_name,
                            yattr,
                            test_size,
                            savef,
                            use_pipeline=use_pipeline,
                            scale_X=scale_X,
                            **sk,
                        )
                    if lcen_kwargs is not None:
                        self.do_LCEN(attr_name, yattr, test_size, savef, **lcen_kwargs)

    def check_dir_exists(self, directory: str):
        """Check if a directory exists, and if not, create it.

        Parameters
        ----------
        directory : str
            path to directory
        """
        try:
            os.makedirs(directory)
        except OSError:
            pass

    def plot_df_heatmap(self, df, fig=None, **plotly_kwargs):
        """Plot the heatmap of a dataframe. Used most often to visualize best fit scores of various models.

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        fig : _type_, optional, default None
            plotly figure for plotting. If None, will create a new figure.
        """
        if fig == None:
            fig = go.Figure()

        fig.add_trace(go.Heatmap(x=df.columns, y=df.index, z=df, **plotly_kwargs))
        fig.update_traces(colorbar=dict(title="Score"))
        fig.show()

    def tsfresh_feature_selection(self, feat, yattr, fdr_level=0.05):
        """Use tsfresh to downselect features before fitting a model.

        Parameters
        ----------
        feat : str
            Feature to select features from.
        yattr : str
            Target variable to select features for.
        fdr_level : float, optional, default=0.05
            FDR level to use for feature selection.

        Returns
        -------
        selected_features : list
            List of selected features.
        """

        if "select_features" not in globals():
            from tsfresh import select_features
        X = getattr(self, feat)
        y = getattr(self, yattr)
        features_df = pd.DataFrame(X)
        selected_features = select_features(features_df, y, fdr_level=fdr_level)
        selected_feature_ids = np.array(selected_features.columns)
        setattr(
            self,
            feat + "_tsfresh_selected",
            features_df[selected_feature_ids].to_numpy(),
        )
        setattr(self, feat + "_tsfresh_selected_feature_ids", selected_feature_ids)
        self.add_generated_feature(feat + "_tsfresh_selected")

    def pearson_feature_selection(
        self, yattr, feature_list=[], k=3, return_top_k=False, return_vals=False
    ):
        """Use Pearson's correlation coefficient to identify
        top k linearly correlated features.

        Parameters
        ----------
        yattr : str
            Target variable to select features for.
        feature_list : list, optional, default=[]
            List of features to select from. If empty, will select from all features.
        k : int, optional, default=3
            Number of top features to select. Currently, ordering features based on the max
            pearson's correlation coefficient for a given feature array.
        return_top_k : bool, optional, default=False
            If True, will return the top k features.
        """
        if "r_regression" not in globals():
            from sklearn.feature_selection import r_regression

        if feature_list == []:
            feature_list = self.generated_features

        max_pearsons = []
        all_pearsons = {}
        for feat in tqdm(feature_list):
            feat_arr = getattr(self, feat)
            pearsons = r_regression(feat_arr, getattr(self, yattr))
            all_pearsons[feat] = pearsons
            pearsons = np.abs(pearsons)
            max_pearsons.append(np.max(pearsons))
        top_k_pearsons = np.argsort(max_pearsons)[::-1][:k]
        top_k_features = np.array(feature_list)[top_k_pearsons]
        setattr(self, yattr + "_pearson_selected_features", list(top_k_features))
        if return_top_k:
            return top_k_features
        if return_vals:
            return all_pearsons

    def mutual_info_feature_selection(
        self, yattr, feature_list=[], k=3, return_top_k=False, return_vals=False
    ):
        """Use mutual information to identify top k features most correlated with the target variable.

        Parameters
        ----------
        yattr : str
            Target variable to select features for.
        feature_list : list, optional, default=[]
            List of features to select from. If empty, will select from all features.
        k : int, optional, default=3
            Number of top features to select.
        return_values : bool, optional, default=False
            If True, will return the values of the top k features.
        """
        if "mutual_info_regression" not in globals():
            from sklearn.feature_selection import mutual_info_regression
        if feature_list == []:
            feature_list = self.generated_features
        max_mutual_infos = []
        all_mutual_infos = {}
        for feat in tqdm(feature_list):
            feat_arr = getattr(self, feat)
            mutual_infos = mutual_info_regression(feat_arr, getattr(self, yattr))
            max_mutual_infos.append(np.max(mutual_infos))
            all_mutual_infos[feat] = mutual_infos
        top_k_mutual_infos = np.argsort(max_mutual_infos)[::-1][:k]
        top_k_features = np.array(feature_list)[top_k_mutual_infos]
        setattr(self, yattr + "_MIF_selected_features", list(top_k_features))

        if return_top_k:
            return top_k_features
        if return_vals:
            return all_mutual_infos

    def maximal_correlation_feature_selection(
        self, yattr, feature_list=[], k=3, return_top_k=False, return_vals=False
    ):
        """Use maximal correlation to identify top k features most correlated with the target variable.

        Parameters
        ----------
        yattr : str
            Target variable to select features for.
        """
        if "r_regression" not in globals():
            from sklearn.feature_selection import r_regression
        if "model" not in globals():
            from ace import model

        if feature_list == []:
            feature_list = self.generated_features
            print(
                "WARNING : running on all features,this can take ~2 min per feature to run"
            )

        max_maximal_correlations = []
        all_maximal_correlations = {}
        for feat in tqdm(feature_list):
            feat_arr = getattr(self, feat)
            maximal_correlations = maximal_correlation_from_ace(
                feat_arr, getattr(self, yattr)
            )
            max_maximal_correlations.append(maximal_correlations)
            all_maximal_correlations[feat] = maximal_correlations
        top_k_maximal_correlations = np.argsort(max_maximal_correlations)[::-1][:k]
        top_k_features = np.array(feature_list)[top_k_maximal_correlations]
        setattr(
            self, yattr + "_maximal_correlation_selected_features", list(top_k_features)
        )
        if return_top_k:
            return top_k_features
        if return_vals:
            return all_maximal_correlations

    def auto_feature_generation(self, feature_generation_dict={}):
        """Generate features for all loaded spectra.

        Parameters
        ----------
        feature_generation_dict : dict, optional
            Mapping of ``{specname: {feature_method: kwargs}}``.
            If empty (default), generates all available features with default
            parameters for every spectrum in ``self.specnames``.
        """
        if feature_generation_dict == {}:
            print(
                "generating features on all input spectra for all available features with default ranges"
            )
            for spec in self.specnames:
                if not hasattr(self, spec + "_xaxis"):
                    print(
                        f"warning, {spec} does not have x-axis, adding x-axis that is just np.arange(len({spec}))"
                    )
                    self.add_x_axis(spec, np.arange(getattr(self, spec).shape[1]))
                feature_generation_dict[spec] = {
                    "pointwise_features": {"xrange": getattr(self, spec + "_xaxis")},
                    "polynomial_featuresall": {},
                    "featureagg": {"n_clusters": 15},
                    "pca": {"n_components": 15},
                    "cdf": {},
                    "cwt": {},
                    "fft_features": {"xrange": getattr(self, spec + "_xaxis")},
                }
        for spec in feature_generation_dict:
            if not hasattr(self, spec + "_xaxis"):
                print(
                    f"warning, {spec} does not have x-axis, adding x-axis that is just np.arange(len({spec}))"
                )
                self.add_x_axis(spec, np.arange(getattr(self, spec).shape[1]))
            for feat in feature_generation_dict[spec]:
                kwargs = feature_generation_dict[spec][feat]
                self.add_spectral_feature(spec, feat, **kwargs)

    def auto_ss(
        self,
        yattr,
        auto_dict={"feature_generation": {}, "RF": {}, "LCEN": {}},
        do_ace=False,
        k=3,
    ):
        """Automatic feature selection

        Parameters
        ----------
        yattr : str
            Target variable to select features for.

        auto_dict : dict, optional, default empty dictionary for 'feature_generation', 'RF', and 'LCEN'
            Dictionary for specifics on the features to generate and model cross-validation.
            Defaults to generating all available features, and using default hyperparameter grids for cross validation.
            for spectra + feature pairs.
        do_ace : bool, optional, default = False
            Whether or not to use the ace package to do feature selection with
            maximal correlation analysis. Thus far seems to give the same results as
            pearson's r. Also can take a long time
        k : int, optional, default=3
            Number of top features to select for moddel training.
        """
        if auto_dict["RF"] == {}:
            print(
                "0.a. No model hyperparameter grid input for random forest, using default hyperparams for cross validation."
            )
            auto_dict["RF"] = {
                "estimator": RandomForestRegressor(),
                "param_grid": {
                    "n_estimators": [50, 100, 200, 300],
                    "criterion": ["squared_error"],
                    "max_depth": [10, 35, None],
                },
                "refit": True,
                "metric": mean_squared_error,
            }

        if auto_dict["LCEN"] == {}:
            print(
                "0.b. No model hyperparameter grid input for LCEN, using default hyperparams for cross validation."
            )
            auto_dict["LCEN"] = {
                "l1_ratio": [0.01, 0.1, 0.3, 0.5, 0.6, 0.99],
                "K_fold": 5,
                "cv_method": "Kfold",
                "alpha": 5,
                "degree": [1],
                "trans_type": "poly",
                "LCEN_cutoff": 4e-2,
                "LCEN_interaction": False,
                "scale_X": True,
                "scale_y": True,
                "verbosity_level": 1,
            }

        print("1. Generating Features")
        feature_generation_dict = auto_dict["feature_generation"]
        self.auto_feature_generation(feature_generation_dict)
        print("Feature generation Done ✅")

        print(f"2. Selecting {k} Features to build each model on")
        feature_list = self.generated_features
        if do_ace:
            print("Running Maximum Correlation Analysis")
            ace_selected = self.maximal_correlation_feature_selection(
                yattr, feature_list, k, return_top_k=True
            )
            print(f"Maximum Correlation Analysis selected {ace_selected}")
        else:
            ace_selected = []
        print("Running Mutual Information Analysis")
        mif_selected = self.mutual_info_feature_selection(
            yattr, feature_list, k, return_top_k=True
        )
        print("Running Pearson's Correlation Analysis")
        r_selected = self.pearson_feature_selection(
            yattr, feature_list, k, return_top_k=True
        )
        print(f"Mutual Information selected {mif_selected}")
        print(f"Pearson's correlation selected {r_selected}")
        print("Feature Selection Done ✅")

        print("3. Running LCEN and RF models")

        for feat in r_selected:
            self.do_LCEN(feat, yattr=yattr, **auto_dict["LCEN"])

        for feat in mif_selected:
            self.do_sk(feat, yattr, **auto_dict["RF"])

        for feat in ace_selected:
            self.do_sk(feat, yattr, **auto_dict["RF"])
            self.do_lcen(feat, yattr=yattr, **auto_dict["LCEN"])
        print("Model Training Done ✅")
