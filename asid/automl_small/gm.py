from .tools import choose_and_fit_model, check_gen_model_list, check_sim_metric_list, tools_check_number, check_x_y, \
    check_gm_fitted
from .generative_model_estimation import fit_model
from sklearn.preprocessing import StandardScaler
from .generative_model_sampling import get_sampled_data
from .dataset_similarity_metrics import calc_metrics
import random
import numpy as np
from datetime import datetime
from numpy import ndarray
from typing import Union


class GenerativeModel(object):
    """
    GenerativeModel is a tool designed to find an appropriate generative model for small tabular data. It estimates the
    similarity of synthetic samples, accounts for overfitting and outputs the optimal option.

    Parameters
    ----------
    gen_model_type : {"optimize", "sklearn_kde", "stats_kde_cv_ml", "stats_kde_cv_ls", "gmm", "bayesian_gmm", "ctgan",
        "copula", "copulagan", "tvae"}, default="optimize"
        An "optimize" option refers to the process of choosing the optimal generative model with regard to the
        overfitting or a specific type of generative model could be chosen.

    similarity_metric : {"zu", "c2st_acc"} or None, default="zu"
        Metric that is used to choose the optimal generative model. "zu" metric refers to a Data-Copying Test from (C.
        Meehan et al., 2020). "c2st_acc" refers to a Classifier Two-Sample Test, that uses a 1-Nearest Neighbor
        classifier and computes the leave-one-out (LOO) accuracy separately for the real and generated samples (Q. Xu et
        al., 2018).

    num_syn_samples : int, default=100
        The number of synthetic samples generated to evaluate the similarity_metric score.

    hyperopt_time : int, default=0
        The runtime setting (in seconds) for Hyperopt optimization. Hyperopt is used to find the optimal
        hyper-parameters for generative models except for "stats_kde_cv_ml", "stats_kde_cv_ls", "copula" methods.

    Attributes
    ----------
    gen_model_ : instance
        Fitted generative model.

    gen_model_label_ : instance
        Generative algorithm label.

    score_ : float
        Mean value of similarity_metric for the optimal generative model.

    scaler_ : instance
        Fitted scaler that is applied prior to generative model estimation.

    info_ : dict
        Score and time data series for the range of estimated generative models.

    References
    ----------
    Meehan C., Chaudhuri K., Dasgupta S. (2020) “A non-parametric test to detect data-copying in generative models”
    International Conference on Artificial Intelligence and Statistics.

    Xu, Q., Huang, G., Yuan, Y., Guo, C., Sun, Y., Wu, F., & Weinberger, K. (2018) “An empirical study on evaluation
    metrics of generative adversarial networks” arXiv preprint arXiv:1806.07755.
    """

    def __init__(self, gen_model_type="optimize", similarity_metric="zu", num_syn_samples=100, hyperopt_time=0):
        check_gen_model_list(gen_model_type)
        check_sim_metric_list(similarity_metric, gen_model_type)
        tools_check_number(num_syn_samples, int, "positive")
        tools_check_number(hyperopt_time, int, "non-negative")
        self.gen_model_type = gen_model_type
        self.similarity_metric = similarity_metric
        self.num_syn_samples = num_syn_samples
        self.hyperopt_time = hyperopt_time
        self.gen_model_ = None
        self.gen_model_label_ = None
        self.score_ = None
        self.scaler_ = None
        self.info_ = {}

    def fit(self, data: ndarray):
        """
        Fits GenerativeModel instance.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Training sample.

        Returns
        -------
        self : GenerativeModel instance
            Fitted generative model.
        """
        check_x_y(data)
        scaler = StandardScaler()
        scaler.fit(data)
        self.scaler_ = scaler
        data_scaled = scaler.transform(data)
        if self.gen_model_type == "optimize":
            t0 = datetime.now()
            gen_model, gen_alg_label, score, log_dict = choose_and_fit_model(data, self.similarity_metric, scaler,
                                                                             data_scaled, self.num_syn_samples,
                                                                             self.hyperopt_time)
            self.gen_model_ = gen_model
            self.gen_model_label_ = gen_alg_label
            self.score_ = score
            self.info_ = {"gen_models": log_dict}
            print("The best generative model is " + self.gen_model_label_)
            print(self.similarity_metric + " metric: " + str(self.score_))
            print("Training time: ", datetime.now() - t0)
        else:
            t0 = datetime.now()
            self.gen_model_ = fit_model(self.gen_model_type, data_scaled, self.hyperopt_time)
            self.gen_model_label_ = self.gen_model_type
            print(self.gen_model_type + " model is fitted.")
            print("Training time: ", datetime.now() - t0)
        return self

    def sample(self, sample_size: int, random_state=42) -> ndarray:
        """
        Generates synthetic sample from GenerativeModel.

        Parameters
        ----------
        sample_size : int
            Required sample size.

        random_state : int
            Random state.

        Returns
        -------
        sampled_data : array-like of shape (n_samples, n_features)
            Synthetic sample.
        """
        check_gm_fitted(self)
        tools_check_number(sample_size, int, "positive")
        tools_check_number(random_state, int, "positive")
        sampled_data = get_sampled_data(self.gen_model_, sample_size, [random_state],
                                        self.gen_model_label_, self.scaler_)[0]
        print("Synthetic sample is generated. The shape of sampled dataset: ", sampled_data.shape)
        return sampled_data

    def score(self, train_data: ndarray, similarity_metric: str = "zu", test_data: Union[None, ndarray] = None) -> \
            Union[float, dict]:
        """
        Evaluates the similarity of GenerativeModel samples and train data with the specified similarity metric.

        Parameters
        ----------
        train_data : array-like of shape (n_samples, n_features)
            Training sample.

        test_data : array-like of shape (n_samples, n_features)
            Test sample for "zu" calculation.

        similarity_metric : {"zu", "c2st_acc", "roc_auc", "ks_test"}, default="zu"
            Metric that is used to choose the optimal generative model. "zu" metric refers to a Data-Copying Test from
            (C. Meehan et al., 2020). "c2st_acc" refers to a Classifier Two-Sample Test, that uses a 1-Nearest Neighbor
            classifier and computes the leave-one-out (LOO) accuracy separately for the real and generated samples (Q.
            Xu et al., 2018). "roc_auc" refers to ROC AUC for gradient boosting classifier (Lopez-Paz, D., & Oquab, M.,
            2017). "ks_test": the marginal distributions of samples are compared using Kolmogorov-Smirnov test (Massey
            Jr, F. J., 1951).

        Returns
        -------
        res_score : float or dict
            Mean value of similarity_metric. For "ks_test" dictionary is output with statistic and p-value resulting
            from permutation test.

        References
        ----------
        Meehan C., Chaudhuri K., Dasgupta S. (2020) “A non-parametric test to detect data-copying in generative models”
        International Conference on Artificial Intelligence and Statistics.

        Xu, Q., Huang, G., Yuan, Y., Guo, C., Sun, Y., Wu, F., & Weinberger, K. (2018) “An empirical study on evaluation
        metrics of generative adversarial networks” arXiv preprint arXiv:1806.07755.

        Lopez-Paz, D., & Oquab, M. (2017) “Revisiting classifier two-sample tests” International Conference on Learning
        Representations.

        Massey Jr, F. J. (1951) “The Kolmogorov-Smirnov test for goodness of fit” Journal of the American statistical
        Association, 46(253): 68-78.
        """
        check_gm_fitted(self)
        check_x_y(train_data)
        if test_data is not None:
            check_x_y(test_data)
        else:
            if similarity_metric == "zu":
                raise ValueError("Test data is required for zu calculation.")
        check_sim_metric_list(similarity_metric, "score")
        random.seed(42)
        seed_val = random.sample(list(range(100000)), self.num_syn_samples)
        sampled_data = get_sampled_data(self.gen_model_, train_data.shape[0], seed_val,
                                        self.gen_model_label_, self.scaler_)
        if similarity_metric in ["ks_test"]:
            score_list = []
            p_val = []
            for sd in sampled_data:
                score = calc_metrics(train_data, sd, similarity_metric)
                score_list.append(score[0])
                p_val.append(score[1])
            p_val = np.array(p_val)
            res_score = {"statistic": np.mean(score_list), "p-value": len(p_val[p_val < 0.05]) / len(p_val)}
        else:
            score_list = []
            for sd in sampled_data:
                score_list.append(calc_metrics(train_data, sd, similarity_metric, test_data))
            res_score = np.mean(score_list)
        print(similarity_metric + " metric = " + str(res_score))
        return res_score
