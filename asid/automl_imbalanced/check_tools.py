import numpy as np
from typing import Any
from asid import utils


def check_num_type(x: Any, num_type: type, num_cl: str):
    """
    Validates the type and classification of a numeric parameter.

    Parameters:
    - x (Any): The value to check.
    - num_type (type): The expected numeric type (e.g., int, float).
    - num_cl (str): The classification of the number, such as "positive", "non-negative", or "negative".

    Raises:
    - TypeError: If the parameter is not of the specified numeric type.
    - ValueError: If the parameter does not match the specified classification.
    """

    return utils.check_num_type(x, num_type, num_cl)


def check_x_y(x: Any, y=None):
    """
    Validates the input features (x) and optionally the target values (y).

    Parameters:
    - x (Any): The input data, expected to be a numpy array.
    - y (optional): The target data, expected to have the same number of samples as x.

    Raises:
    - TypeError: If x is not a numpy array.
    - ValueError: If x or y is empty or if x and y have mismatched sample sizes.
    """

    if not isinstance(x, np.ndarray):
        raise TypeError("X should be an array-like type.")
    if x.size == 0:
        raise ValueError("The dataset has no samples.")
    if y is not None:
        if hasattr(y, "size"):
            if y.size == 0:
                raise ValueError("The dataset has no samples.")
        else:
            if len(y) == 0:
                raise ValueError("The dataset has no samples.")
        if x.shape[0] != len(y):
            raise ValueError("The X and y contain different number of samples.")


def check_abb_fitted(self):
    """
    Checks if the AutoBalanceBoost model is fitted.

    Raises:
    - ValueError: If the model is not fitted.
    """

    if not self.ensemble_:
        raise ValueError("AutoBalanceBoost is not fitted.")


def check_ilc_fitted(self):
    """
    Checks if the ImbalancedLearningClassifier is fitted.

    Raises:
    - ValueError: If the classifier is not fitted.
    """

    if not self.classifer_:
        raise ValueError("ImbalancedLearningClassifier is not fitted.")


def check_eval_metric_list(metric: str):
    """
    Validates the evaluation metric against a predefined list of supported metrics.

    Parameters:
    - metric (str): The evaluation metric to check.

    Raises:
    - ValueError: If the metric is not in the supported list.
    """

    if metric not in ["accuracy", "roc_auc", "log_loss", "f1_macro", "f1_micro", "f1_weighted"]:
        raise ValueError("Metric " + str(metric) + " is not implemented.")
