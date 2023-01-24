"""
This module contains AutoBalanceBoost class.
"""

from .tools_abb import boosting_of_bagging_procedure, get_pred, get_pred_proba, get_feat_imp
from .check_tools import check_num_type, check_x_y, check_abb_fitted
from numpy import ndarray


class AutoBalanceBoost(object):
    """
    AutoBalanceBoost classifier is a tailored imbalanced learning framework
    with the built-in hyper-parameters tuning procedure.

    Parameters
    ----------
    num_iter : int, default=40
        The number of boosting iterations.

    num_est : int, default=16
        The number of estimators in the base ensemble.

    Attributes
    ----------
    ensemble_ : list
        The list of fitted ensembles that constitute AutoBalanceBoost model.

    param_ : dict
        The optimal values of AutoBalanceBoost hyper-parameters.
    """

    def __init__(self, num_iter=40, num_est=16):
        check_num_type(num_iter, int, "positive")
        check_num_type(num_est, int, "positive")
        self.num_iter = num_iter
        self.num_est = num_est
        self.ensemble_ = None
        self.param_ = {}

    def fit(self, X: ndarray, y: ndarray):
        """
        Fits AutoBalanceBoost model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training sample.

        y : array-like
            The target values.

        Returns
        -------
        self : AutoBalanceBoost classifier
            Fitted estimator.
        """
        check_x_y(X, y)
        self.ensemble_, self.param_ = boosting_of_bagging_procedure(X, y, self.num_iter, self.num_est)
        return self

    def predict(self, X: ndarray) -> ndarray:
        """
        Predicts class label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test sample.

        Returns
        -------
        pred : array-like
            The predicted class.
        """
        check_abb_fitted(self)
        check_x_y(X)
        pred = get_pred(self.ensemble_, X)
        return pred

    def predict_proba(self, X: ndarray) -> ndarray:
        """
        Predicts class probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test sample.

        Returns
        -------
        pred_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        check_abb_fitted(self)
        check_x_y(X)
        pred_proba = get_pred_proba(self.ensemble_, X)
        return pred_proba

    def feature_importances(self) -> ndarray:
        """
        Calculates normalized feature importances.

        Returns
        -------
        feat_imp : array-like
            The normalized feature importances.
        """
        check_abb_fitted(self)
        feat_imp = get_feat_imp(self.ensemble_)
        return feat_imp
