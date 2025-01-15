from .tools_abb import boosting_of_bagging_procedure, get_pred, get_pred_proba, get_feat_imp
from .check_tools import check_tools_verify_number, check_x_y, check_abb_fitted
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
        check_tools_verify_number(num_iter, int, "positive")
        check_tools_verify_number(num_est, int, "positive")
        self.num_iter = num_iter
        self.num_est = num_est
        self.ensemble_ = None
        self.param_ = {}

    def fit(self, x: ndarray, y: ndarray):
        """
        Fits AutoBalanceBoost model.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training sample.

        y : array-like
            The target values.

        Returns
        -------
        self : AutoBalanceBoost classifier
            Fitted estimator.
        """
        check_x_y(x, y)
        self.ensemble_, self.param_ = boosting_of_bagging_procedure(x, y, self.num_iter, self.num_est)
        return self

    def predict(self, x: ndarray) -> ndarray:
        """
        Predicts class label.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Test sample.

        Returns
        -------
        pred : array-like
            The predicted class.
        """
        check_abb_fitted(self)
        check_x_y(x)
        pred = get_pred(self.ensemble_, x)
        return pred

    def predict_proba(self, x: ndarray) -> ndarray:
        """
        Predicts class probability.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Test sample.

        Returns
        -------
        pred_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        check_abb_fitted(self)
        check_x_y(x)
        pred_proba = get_pred_proba(self.ensemble_, x)
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
