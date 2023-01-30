import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sdv.tabular import GaussianCopula, CTGAN, CopulaGAN, TVAE
import torch
from sklearn.model_selection import KFold
from sdv.evaluation import evaluate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.neighbors import KernelDensity
from hyperopt import fmin, tpe, space_eval
from hyperopt import hp
import warnings
from numpy import ndarray

warnings.filterwarnings("ignore")


def calc_gmm_acc(params: dict, data: ndarray) -> float:
    """
    Estimates the performance of GaussianMixture model with hyper-parameters values.

    Parameters
    ----------
    params : dict
        Hyper-parameters values.

    data : array-like of shape (n_samples, n_features)
        Training sample.

    Returns
    -------
    score : float
        Performance score.
    """
    skf = KFold(n_splits=3, random_state=42, shuffle=True)
    res_metric = []
    for index in skf.split(data):
        x_train, x_test = data[index[0], :], data[index[1], :]
        model = GaussianMixture(n_components=int(params["n_components"]), covariance_type=params['covariance_type'],
                                random_state=42)
        model.fit(x_train)
        res_metric.append(model.score(x_test))
    score = -np.mean(res_metric)
    return score


def get_gmm_model(data: ndarray, hyp_time: int) -> object:
    """
    Estimates GMM model.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    hyp_time : int
        The runtime setting (in seconds) for Hyperopt optimization.

    Returns
    -------
    gmm : instance
        Fitted generative model.
    """
    if hyp_time == 0:
        gmm = GaussianMixture(random_state=42)
        gmm.fit(data)
    else:
        space = {"n_components": hp.quniform("n_components", 1, int(round(data.shape[0] / 3 * 2) - 1), 1),
                 "covariance_type": hp.choice("covariance_type", ['full', 'tied', 'diag', 'spherical'])}
        best = fmin(fn=lambda params: calc_gmm_acc(params, data), space=space,
                    algo=tpe.suggest, timeout=hyp_time, rstate=np.random.seed(42))
        best = space_eval(space, best)
        gmm = GaussianMixture(n_components=int(best["n_components"]), covariance_type=best['covariance_type'],
                              random_state=42)
        gmm.fit(data)
    return gmm


def calc_bayesian_gmm_acc(params: dict, data: ndarray) -> float:
    """
    Estimates the performance of BayesianGaussianMixture model with hyper-parameters values.

    Parameters
    ----------
    params : dict
        Hyper-parameters values.

    data : array-like of shape (n_samples, n_features)
        Training sample.

    Returns
    -------
    score : float
        Performance score.
    """
    skf = KFold(n_splits=3, random_state=42, shuffle=True)
    res_metric = []
    for index in skf.split(data):
        x_train, x_test = data[index[0], :], data[index[1], :]
        model = BayesianGaussianMixture(n_components=int(params["n_components"]),
                                        covariance_type=params['covariance_type'],
                                        weight_concentration_prior_type=params['weight_concentration_prior_type'],
                                        weight_concentration_prior=params['weight_concentration_prior'],
                                        random_state=42)
        model.fit(x_train)
        res_metric.append(model.score(x_test))
    score = -np.mean(res_metric)
    return score


def get_bayesian_gmm_model(data: ndarray, hyp_time: int) -> object:
    """
    Estimates Bayesian GMM model.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    hyp_time : int
        The runtime setting (in seconds) for Hyperopt optimization.

    Returns
    -------
    gmm : instance
        Fitted generative model.
    """
    if hyp_time == 0:
        gmm = BayesianGaussianMixture(random_state=42)
        gmm.fit(data)
    else:
        space = {"n_components": hp.quniform("n_components", 1, int(round(data.shape[0] / 3 * 2) - 1), 1),
                 "covariance_type": hp.choice("covariance_type", ['full', 'tied', 'diag', 'spherical']),
                 "weight_concentration_prior_type": hp.choice("weight_concentration_prior_type",
                                                              ['dirichlet_process', 'dirichlet_distribution']),
                 'weight_concentration_prior': hp.uniform('weight_concentration_prior', 0.0001, 10000)}
        best = fmin(fn=lambda params: calc_bayesian_gmm_acc(params, data), space=space,
                    algo=tpe.suggest, timeout=hyp_time, rstate=np.random.seed(42))
        best = space_eval(space, best)
        gmm = BayesianGaussianMixture(n_components=int(best["n_components"]), covariance_type=best['covariance_type'],
                                      weight_concentration_prior_type=best['weight_concentration_prior_type'],
                                      weight_concentration_prior=best['weight_concentration_prior'], random_state=42)
        gmm.fit(data)
    return gmm


def get_copula_model(data: ndarray) -> object:
    """
    Estimates Copula model.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    Returns
    -------
    model : instance
        Fitted generative model.
    """
    np.random.seed(42)
    model = GaussianCopula()
    data = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    model.fit(data)
    return model


def calc_sdv_acc(params: dict, data: ndarray, alg: str) -> float:
    """
    Estimates the performance of SDV model with hyper-parameters values.

    Parameters
    ----------
    params : dict
        Hyper-parameters values.

    data : array-like of shape (n_samples, n_features)
        Training sample.

    alg : str
        Algorithm label.

    Returns
    -------
    score : float
        Performance score.
    """
    cv_rs = 42
    np.random.seed(42)
    torch.manual_seed(42)
    skf = KFold(n_splits=3, shuffle=True, random_state=cv_rs)
    res_metric = []
    for index in skf.split(data):
        x_train, x_test = data.iloc[index[0]], data.iloc[index[1]]
        if alg == "ctgan":
            model = CTGAN(batch_size=int(params["batch_size"]), cuda=False)
        elif alg == "copulagan":
            model = CopulaGAN(batch_size=int(params["batch_size"]), cuda=False)
        elif alg == "tvae":
            model = TVAE(batch_size=int(params["batch_size"]), cuda=False)
        model.fit(x_train)
        sub_metric_list = []
        for i in range(10):
            sampled_data = model.sample(len(x_test))
            sub_metric_list.append(evaluate(sampled_data, x_test))
        res_metric.append(np.mean(sub_metric_list))
    score = -np.mean(res_metric)
    return score


def get_ctgan_model(data: ndarray, hyp_time: int) -> object:
    """
    Estimates CTGAN model.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    hyp_time : int
        The runtime setting (in seconds) for Hyperopt optimization.

    Returns
    -------
    model : instance
        Fitted generative model.
    """
    if hyp_time < 15:
        data = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
        model = CTGAN(cuda=False)
        model.fit(data)
    else:
        space = {"batch_size": hp.quniform("batch_size", 300, 800, 50)}
        data = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
        best = fmin(fn=lambda params: calc_sdv_acc(params, data, "ctgan"), space=space,
                    algo=tpe.suggest, timeout=hyp_time, rstate=np.random.seed(42))
        best = space_eval(space, best)
        model = CTGAN(batch_size=int(best["batch_size"]), cuda=False)
        model.fit(data)
    return model


def get_copulagan_model(data: ndarray, hyp_time: int) -> object:
    """
    Estimates CopulaGAN model.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    hyp_time : int
        The runtime setting (in seconds) for Hyperopt optimization.

    Returns
    -------
    model : instance
        Fitted generative model.
    """
    if hyp_time < 15:
        data = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
        model = CopulaGAN(cuda=False)
        model.fit(data)
    else:
        space = {"batch_size": hp.quniform("batch_size", 300, 800, 50)}
        data = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
        best = fmin(fn=lambda params: calc_sdv_acc(params, data, "copulagan"), space=space,
                    algo=tpe.suggest, timeout=hyp_time, rstate=np.random.seed(42))
        best = space_eval(space, best)
        model = CopulaGAN(batch_size=int(best["batch_size"]), cuda=False)
        model.fit(data)
    return model


def get_tvae_model(data: ndarray, hyp_time: int) -> object:
    """
    Estimates TVAE model

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    hyp_time : int
        The runtime setting (in seconds) for Hyperopt optimization.

    Returns
    -------
    model : instance
        Fitted generative model.
    """
    if hyp_time < 15:
        data = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
        model = TVAE(cuda=False)
        model.fit(data)
    else:
        space = {"batch_size": hp.quniform("batch_size", 300, 800, 50)}
        data = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
        best = fmin(fn=lambda params: calc_sdv_acc(params, data, "tvae"), space=space,
                    algo=tpe.suggest, timeout=hyp_time, rstate=np.random.seed(42))
        best = space_eval(space, best)
        model = TVAE(batch_size=int(best["batch_size"]), cuda=False)
        model.fit(data)
    return model


def calc_kde_acc(params: dict, data: ndarray) -> float:
    """
    Estimates the performance of KDE (sklearn implementation) with hyper-parameters values.

    Parameters
    ----------
    params : dict
        Hyper-parameters values.

    data : array-like of shape (n_samples, n_features)
        Training sample.

    Returns
    -------
    score : float
        Performance score.
    """
    skf = KFold(n_splits=3, random_state=42, shuffle=True)
    res_metric = []
    for index in skf.split(data):
        x_train, x_test = data[index[0], :], data[index[1], :]
        model = KernelDensity(kernel="gaussian", bandwidth=params['bandwidth'])
        model.fit(x_train)
        res_metric.append(model.score(x_test))
    score = -np.mean(res_metric)
    return score


def sklearn_kde(data: ndarray, hyp_time: int) -> object:
    """
    Estimates KDE (sklearn implementation).

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    hyp_time : int
        The runtime setting (in seconds) for Hyperopt optimization.

    Returns
    -------
    kde : instance
        Fitted generative model.
    """
    if hyp_time == 0:
        kde = KernelDensity(kernel="gaussian")
        kde.fit(data)
    else:
        space = {'bandwidth': hp.uniform('bandwidth', 0.01, 2)}
        best = fmin(fn=lambda params: calc_kde_acc(params, data), space=space,
                    algo=tpe.suggest, timeout=hyp_time, rstate=np.random.seed(42))
        best = space_eval(space, best)
        kde = KernelDensity(kernel="gaussian", bandwidth=best['bandwidth'])
        kde.fit(data)
    return kde


def stats_kde(data: ndarray, method: str) -> object:
    """
    Estimates KDE (Statsmodels).

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.

    method : {"cv_ml", "cv_ls"}
        CV type for bandwidth selection.

    Returns
    -------
    kde : instance
        Fitted generative model.
    """
    var_type = "c" * data.shape[1]
    kde = KDEMultivariate(data, var_type, bw=method)
    return kde


def fit_model(gen_algorithm: str, data: ndarray, hyp_time: int) -> object:
    """
    Fits generative model.

    Parameters
    ----------
    gen_algorithm : {"sklearn_kde", "stats_kde_cv_ml", "stats_kde_cv_ls", "gmm", "bayesian_gmm", "ctgan",
        "copula", "copulagan", "tvae"}
        Generative algorithm label.

    data : array-like of shape (n_samples, n_features)
        Training sample.

    hyp_time : int
        The runtime setting (in seconds) for Hyperopt optimization.

    Returns
    -------
    model : instance
        Fitted generative model.
    """
    if gen_algorithm == "sklearn_kde":
        model = sklearn_kde(data, hyp_time)
    elif gen_algorithm in ["stats_kde_cv_ml", "stats_kde_cv_ls"]:
        method_list = gen_algorithm.split("_")
        method_name = method_list[2] + "_" + method_list[3]
        model = stats_kde(data, method_name)
    elif gen_algorithm == "gmm":
        model = get_gmm_model(data, hyp_time)
    elif gen_algorithm == "bayesian_gmm":
        model = get_bayesian_gmm_model(data, hyp_time)
    elif gen_algorithm == "ctgan":
        model = get_ctgan_model(data, hyp_time)
    elif gen_algorithm == "copula":
        model = get_copula_model(data)
    elif gen_algorithm == "copulagan":
        model = get_copulagan_model(data, hyp_time)
    elif gen_algorithm == "tvae":
        model = get_tvae_model(data, hyp_time)
    return model
