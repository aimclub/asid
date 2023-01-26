"""
This module contains ImbalancedLearningClassifier class.
"""

from .tools_ilc import choose_and_fit_ilc, calc_leaderboard
from .check_tools import check_num_type, check_eval_metric_list, check_x_y, check_ilc_fitted
from datetime import datetime
from sklearn import preprocessing
from numpy import ndarray


class ImbalancedLearningClassifier(object):
    """
    ImbalancedLearningClassifier finds an optimal classifier among the combinations of balancing
    procedures from imbalanced-learn library (with Hyperopt optimization) and state-of-the-art ensemble classifiers,
    and the tailored classifier AutoBalanceBoost.

    Parameters
    ----------
    split_num : int, default=5
        The number of splitting iterations for obtaining an out-of-fold score. If the number is a 5-fold, then
        StratifiedKFold with 5 splits is repeated with the required number of seeds, otherwise StratifiedShuffleSplit
        with split_num splits is used.

    hyperopt_time : int, default=0
        The runtime setting (in seconds) for Hyperopt optimization. Hyperopt is used to find the optimal
        hyper-parameters for balancing procedures.

    eval_metric : {"accuracy", "roc_auc", "log_loss", "f1_macro", "f1_micro", "f1_weighted"}, default="f1_macro"
        Metric that is used to evaluate the model performance and to choose the best option.

    Attributes
    ----------
    classifer_ : instance
        Optimal fitted classifier.

    classifer_label_ : str
        Optimal classifier label.

    score_ : float
        Averaged out-of-fold value of eval_metric for the optimal classifier.

    scaler_ : instance
        Fitted scaler that is applied prior to classifier estimation.

    encoder_ : instance
        Fitted label encoder.

    classes_ : array-like
        Class labels.

    evaluated_models_scores_ : dict
        Score series for the range of estimated classifiers.

    evaluated_models_time_ : dict
        Time data for the range of estimated classifiers.

    conf_int_ : tuple
        95% confidence interval for the out-of-fold value of eval_metric for the optimal classifier.
    """

    def __init__(self, split_num=5, hyperopt_time=0, eval_metric="f1_macro"):
        check_num_type(split_num, int, "positive")
        check_num_type(hyperopt_time, int, "non-negative")
        check_eval_metric_list(eval_metric)
        self.classifer_ = None
        self.classifer_label_ = None
        self.split_num = split_num
        self.hyperopt_time = hyperopt_time
        self.score_ = None
        self.scaler_ = None
        self.encoder_ = None
        self.classes_ = None
        self.evaluated_models_scores_ = None
        self.evaluated_models_time_ = None
        self.eval_metric = eval_metric
        self.conf_int_ = None

    def fit(self, x: ndarray, y: ndarray):
        """
        Fits ImbalancedLearningClassifier model.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training sample.

        y : array-like
            The target values.

        Returns
        -------
        self : ImbalancedLearningClassifier instance
            Fitted estimator.
        """
        check_x_y(x, y)
        t0 = datetime.now()
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        self.classes_ = le.classes_
        self.encoder_ = le
        self.classifer_, self.classifer_label_, self.score_, self.scaler_, self.evaluated_models_scores_, \
        self.evaluated_models_time_, self.conf_int_ = choose_and_fit_ilc(self, x, y)
        print("The best generative model is " + self.classifer_label_)
        print("Leader " + self.eval_metric + " score: " + str(round(self.score_, 4)))
        print("Fitting time: ", datetime.now() - t0)
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
        check_ilc_fitted(self)
        check_x_y(x)
        if self.classifer_label_ == "AutoBalanceBoost":
            pred = self.classifer_.predict(x)
        else:
            x_scaled = self.scaler_.transform(x)
            pred = self.classifer_.predict(x_scaled)
        pred = self.encoder_.inverse_transform(pred)
        return pred

    def predict_proba(self, x) -> ndarray:
        """
        Predicts class label probability.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Test sample.

        Returns
        -------
        pred_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        check_ilc_fitted(self)
        check_x_y(x)
        if self.classifer_label_ == "AutoBalanceBoost":
            pred_proba = self.classifer_.predict_proba(x)
        else:
            x_scaled = self.scaler_.transform(x)
            pred_proba = self.classifer_.predict_proba(x_scaled)
        return pred_proba

    def leaderboard(self) -> dict:
        """
        Calculates the leaderboard statistics.

        Returns
        -------
        ls : dict
            The leaderboard statistics that includes sorted lists in accordance with the following indicators:
            "Mean score", "Mean rank", "Share of experiments with the first place, %",
            "Average difference with the leader, %".
        """
        check_ilc_fitted(self)
        ls = calc_leaderboard(self)
        print("Leaderboard statistics")
        print("")
        print("Mean score")
        print(ls["Mean score"])
        print("")
        print("Mean rank")
        print(ls["Mean rank"])
        print("")
        print("Share of experiments with the first place, %")
        print(ls["Share of experiments with the first place, %"])
        print("")
        print("Average difference with the leader, %")
        print(ls["Average difference with the leader, %"])
        return ls
