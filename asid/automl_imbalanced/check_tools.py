"""
This module contains functions for exception handling.
"""

import numpy as np
from typing import Any


def check_num_type(x: Any, num_type: type, num_cl: str):
    if isinstance(x, num_type):
        if num_cl == "positive" and x <= 0:
            raise ValueError("The parameter should be " + num_cl + ".")
        elif num_cl == "non-negative" and x < 0:
            raise ValueError("The parameter should be " + num_cl + ".")
        elif num_cl == "negative" and x >= 0:
            raise ValueError("The parameter should be " + num_cl + ".")
    else:
        raise TypeError("The parameter should be of " + str(num_type))


def check_x_y(X: Any, y=None):
    if not isinstance(X, np.ndarray):
        raise TypeError("X should be an array-like type.")
    if X.size == 0:
        raise ValueError("The dataset has no samples.")
    if y is not None:
        if hasattr(y, "size"):
            if y.size == 0:
                raise ValueError("The dataset has no samples.")
        else:
            if len(y) == 0:
                raise ValueError("The dataset has no samples.")
        if X.shape[0] != len(y):
            raise ValueError("The X and y contain different number of samples.")


def check_abb_fitted(self):
    if not self.ensemble_:
        raise ValueError("AutoBalanceBoost is not fitted.")


def check_ilc_fitted(self):
    if not self.classifer_:
        raise ValueError("ImbalancedLearningClassifier is not fitted.")


def check_eval_metric_list(metric: str):
    if metric not in ["accuracy", "roc_auc", "log_loss", "f1_macro", "f1_micro", "f1_weighted"]:
        raise ValueError("Metric " + str(metric) + " is not implemented.")
