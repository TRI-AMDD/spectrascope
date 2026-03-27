"""Fused Lasso regression model implemented with CVXPY.

Provides :class:`FusedLasso`, a scikit-learn compatible regressor that penalizes
successive differences in coefficients (fused penalty), promoting piecewise-constant
solutions — useful for spectral data where important features form contiguous regions.

Requires ``cvxpy`` (optional dependency): ``pip install spectrascope[fused_lasso]``.
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import cvxpy as cp

logger = logging.getLogger(__name__)


class FusedLasso(BaseEstimator, RegressorMixin):
    """Fused lasso regression using CVXPY.

    Solves:

    .. code-block:: text

        minimize  (1 / (2n)) * ||y - (X w + b)||^2
                  + alpha  * sum_j abs(w[j+1] - w[j])
                  + l1_ratio * ||w||_1   (optional)

    Input X is optionally standardized by dividing by max standard deviation
    across features, controlled by ``standardize_X``.

    Parameters
    ----------
    alpha : float, default=1.0
        Strength of the fused penalty on successive differences.
    l1_ratio : float, default=0.0
        Strength of an optional L1 penalty on the coefficients themselves.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    standardize_X : bool, default=True
        Whether to standardize X by dividing by max feature standard deviation before fitting.
    solver_path : str or None, default=None
        CVXPY solver name (e.g. "OSQP", "ECOS", "SCS"). If None, cvxpy chooses.
    """

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.0,
        fit_intercept=True,
        standardize_X=True,
        solver_path=None,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.solver_path = solver_path
        self.standardize_X = standardize_X
        if not self.standardize_X:
            logger.warning("You are not standardizing X")

    def fit(self, X, y):
        """Fit the fused lasso model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : FusedLasso
            Fitted estimator.

        Raises
        ------
        RuntimeError
            If the CVXPY optimization does not converge.
        """
        X = np.asarray(X)
        if self.standardize_X:
            stds = X.std(axis=0)
            self.max_std_ = stds.max()
            X = X / self.max_std_
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
            diffs = w[1:] - w[:-1]  # shape (n_features-1,)
            fused_penalty = cp.norm1(diffs)
        else:
            fused_penalty = 0.0

        # Optional plain L1 penalty on coefficients
        l1_penalty = cp.norm1(w) if self.l1_ratio > 0 else 0.0

        # Total objective
        objective = loss + self.alpha * fused_penalty + self.l1_ratio * l1_penalty

        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(solver_path=self.solver_path)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Optimization failed with status: {prob.status}")

        self.coef_ = w.value.astype(float)
        self.intercept_ = float(b.value) if self.fit_intercept else 0.0

        return self

    def predict(self, X):
        """Predict target values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        X = np.asarray(X)
        if self.standardize_X:
            X = X / self.max_std_
        return X @ self.coef_ + self.intercept_
