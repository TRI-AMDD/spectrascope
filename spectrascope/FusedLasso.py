
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import cvxpy as cp


class FusedLasso(BaseEstimator, RegressorMixin):
    """
    TODO : include standardization 
    Fused lasso regression:
        minimize  (1 / (2n)) * ||y - (X w + b)||_2^2
                  + alpha * sum_j |w_{j+1} - w_j|
                  + l1_ratio * ||w||_1   (optional)

    Parameters
    ----------
    alpha : float, default=1.0
        Strength of the fused penalty on successive differences.

    l1_ratio : float, default=0.0
        Strength of an optional L1 penalty on the coefficients themselves.

    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    solver : str or None, default=None
        CVXPY solver name (e.g. "OSQP", "ECOS", "SCS"). If None, cvxpy chooses.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.0, fit_intercept=True, solver=None, standardize_X = True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.solver = solver        
        self.standardize_X = standardize_X
        if not self.standardize_X: 
            print('You are not standardizing X')

    def fit(self, X, y):
        X = np.asarray(X)  
        if self.standardize_X: 
            stds = X.std(axis=0)
            max_std = stds.max()  
            X = X / max_std
        y = np.asarray(y).ravel()
        n_samples, n_features = X.shape

        # Variables
        w = cp.Variable(n_features)
        if self.fit_intercept:
            b = cp.Variable()
            y_pred = X @ w + b
        else:
            b = None
            y_pred = X @ w

        # Data-fitting term
        loss = (1.0 / (2.0 * n_samples)) * cp.sum_squares(y - y_pred)

        # Fused lasso penalty on differences between successive coefficients
        if n_features > 1:
            diffs = w[1:] - w[:-1]   # shape (n_features-1,)
            fused_penalty = cp.norm1(diffs)
        else:
            fused_penalty = 0.0

        # Optional plain L1 penalty on coefficients
        l1_penalty = cp.norm1(w) if self.l1_ratio > 0 else 0.0

        # Total objective
        objective = loss + self.alpha * fused_penalty + self.l1_ratio * l1_penalty

        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(solver=self.solver)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Optimization failed with status: {prob.status}")

        self.coef_ = w.value.astype(float)
        self.intercept_ = float(b.value) if self.fit_intercept else 0.0

        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
