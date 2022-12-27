"""
This module contains helping functions for GenerativeModel class.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from .generative_model_estimation import fit_model
from .generative_model_sampling import get_sampled_data
from .dataset_similarity_metrics import calc_metrics
from sklearn.model_selection import ShuffleSplit
import random
import math
from datetime import datetime


def choose_and_fit_model(data, similarity_metric, scaler, data_scaled, num_syn_samples, hyp_time):
    """
    Chooses an optimal generative model and fits GenerativeModel instance.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    similarity_metric : {"zu", "c2st_acc"} or None, default="zu"
        Metric that is used to choose the optimal generative model.

    scaler : instance
        Fitted scaler that is applied prior to generative model estimation.

    data_scaled : array-like of shape (n_samples, n_features)
        Normalized training sample.

    num_syn_samples : int
        The number of synthetic samples generated to evaluate the similarity_metric score.
        
    hyp_time : int
        The runtime setting (in seconds) for Hyperopt optimization. Hyperopt is used to find the optimal
        hyper-parameters for generative models.

    Returns
    -------
    gen_model : instance
        Optimal fitted generative model.

    best_alg_label : str
        Optimal generative algorithm label.

    best_score : float
        Mean value of similarity_metric for the optimal generative model.

    log_dict : dict
        Score and time data series for the range of estimated generative models.
    """
    # [suggestion] тут все слилось в одну полосу кода. Выделить бы блоки кода по логическому смыслу, чтобы
    # код читать было лучше
    best_model = None
    best_score = +math.inf
    best_alg_label = None
    random.seed(42)
    seed_val = random.sample(list(range(100000)), num_syn_samples)
    log_dict = {}
    for gen_meth in ["sklearn_kde", "stats_kde_cv_ml", "stats_kde_cv_ls", "gmm", "bayesian_gmm", "ctgan",
                     "copula", "copulagan", "tvae"]:
        log_dict[gen_meth] = {}
        if similarity_metric == "c2st_acc":
            # [suggestion] Со временем можно отдльеный класс выделить таймера, будет что то вроде
            # with Timer() as timer:
            #     gen_model = fit_model(gen_meth, data_scaled, hyp_time)
            #     log_dict[gen_meth]["train_time"] = timer.delta
            # По коду в целом тоже выходит, но выглядит более по блокам. Если интересно - покажу как сделать
            t0 = datetime.now()
            gen_model = fit_model(gen_meth, data_scaled, hyp_time)
            log_dict[gen_meth]["train_time"] = datetime.now() - t0
            t0 = datetime.now()
            sampled_data_list = get_sampled_data(gen_model, data.shape[0], seed_val, gen_meth, scaler)
            log_dict[gen_meth]["sampling_time"] = datetime.now() - t0
            score_list = []
            t0 = datetime.now()
            for sampled_data in sampled_data_list:
                score_list.append(abs(0.5 - calc_metrics(data, sampled_data, similarity_metric)))
            log_dict[gen_meth]["calc_metric_time"] = datetime.now() - t0
        elif similarity_metric == "zu":
            rs = ShuffleSplit(n_splits=1, test_size=0.3, train_size=0.7, random_state=42)
            for indexes in rs.split(data):
                data_train = data[indexes[0], :]
                data_test = data[indexes[1], :]
            train_scaler = StandardScaler()
            train_scaler.fit(data_train)
            data_train_scaled = scaler.transform(data_train)
            t0 = datetime.now()
            gen_model = fit_model(gen_meth, data_train_scaled, hyp_time)
            log_dict[gen_meth]["train_time"] = datetime.now() - t0
            t0 = datetime.now()
            sampled_data_list = get_sampled_data(gen_model, data_train.shape[0], seed_val, gen_meth,
                                                 train_scaler)
            log_dict[gen_meth]["sampling_time"] = datetime.now() - t0
            score_list = []
            t0 = datetime.now()
            for sampled_data in sampled_data_list:
                score_list.append(abs(calc_metrics(data_train, sampled_data, similarity_metric, data_test)))
            log_dict[gen_meth]["calc_metric_time"] = datetime.now() - t0
        score = np.mean(score_list)
        log_dict[gen_meth]["score"] = score_list
        if score < best_score:
            best_score = score
            best_model = gen_model
            best_alg_label = gen_meth
    if similarity_metric == "zu":
        gen_model = fit_model(best_alg_label, data_scaled, hyp_time)
    else:
        gen_model = best_model
    return gen_model, best_alg_label, best_score, log_dict


def check_gen_model_list(metric):
    # [suggestion] gen - generation? Лучше расшифровать, и этот метод используется только в gm. Я бы туда и перенес
    # притом можно сделать его protected, типо "@staticmetod def _check_gen_model_list(metric: str):"
    # А вот ["optimize", "sklearn_kde"...] - тоже бы вынес куда-нибудь в отдельное место
    if metric not in ["optimize", "sklearn_kde", "stats_kde_cv_ml", "stats_kde_cv_ls", "gmm", "bayesian_gmm", "ctgan",
                      "copula", "copulagan", "tvae"]:
        raise ValueError("Generative model " + str(metric) + " is not implemented.")


def check_sim_metric_list(metric, mtype):
    # [suggestion] те же предложения
    if mtype == "optimize":
        if metric not in ["zu", "c2st_acc"]:
            raise ValueError("Metric " + str(metric) + " is not implemented.")
    elif mtype == "score":
        if metric not in ["zu", "c2st_acc", "roc_auc", "ks_test"]:
            raise ValueError("Metric " + str(metric) + " is not implemented.")


def check_num_type(x, num_type, num_cl):
    # [suggestion] num_cl сделать бы Enum, строки быстро и удобно, но со временем сложнее отлаживать и использовать
    if isinstance(x, num_type):
        if num_cl == "positive" and x <= 0:
            raise ValueError("The parameter should be " + num_cl + ".")
        elif num_cl == "non-negative" and x < 0:
            raise ValueError("The parameter should be " + num_cl + ".")
        elif num_cl == "negative" and x >= 0:
            raise ValueError("The parameter should be " + num_cl + ".")
    else:
        raise TypeError("The parameter should be of " + str(num_type))


def check_x_y(X, y=None):
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


def check_gm_fitted(self):
    if not self.gen_model_:
        raise ValueError("GenerativeModel is not fitted.")
