"""
This module contains functions for sampling from fitted generative models.
"""

import numpy as np
import torch
import os
from pathlib import Path


def simple_sample_sklearn_procedure(model, sample_len, seed_list, scaling):
    """
    Sampling synthetic datasets from sklearn KDE.

    Parameters
    ----------
    model : instance
        Fitted generative model.

    sample_len : int
        Synthetic sample size.

    seed_list : list
        The list of random seeds for each synthetic dataset.

    scaling : instance
        Fitted scaler that is applied prior to generative model estimation.

    Returns
    -------
    sampled_data_list : list
        The list with synthetiс datasets.
    """
    sampled_data_list = []
    for seed in seed_list:
        sampled_data = model.sample(n_samples=sample_len, random_state=seed)
        if scaling:
            sampled_data = scaling.inverse_transform(sampled_data)
        sampled_data_list.append(sampled_data)
    return sampled_data_list


def sample_stats(kde, size, seed):
    """
    Base sampling procedure from Statsmodel's KDE.

    Parameters
    ----------
    kde : instance
        Fitted KDE model.

    size : int
        Synthetic sample size.
        
    seed : int
        Random seed.

    Returns
    -------
    sampled_data : array-like of shape (n_samples, n_features)
        Synthetic sample.
    """
    rng = np.random.RandomState(seed)
    n, d = kde.data.shape
    indices = rng.randint(0, n, size)
    cov = np.diag(kde.bw) ** 2
    means = kde.data[indices, :]
    norm = rng.multivariate_normal(np.zeros(d), cov, size)
    sampled_data = np.transpose(means + norm).T
    return sampled_data


def simple_sample_stats_procedure(model, sample_len, seed_list, scaling):
    """
    Sampling synthetic datasets from Statsmodel's KDE.

    Parameters
    ----------
    model : instance
        Fitted generative model.

    sample_len : int
        Synthetic sample size.

    seed_list : list
        The list of random seeds for each synthetic dataset.

    scaling : instance
        Fitted scaler that is applied prior to generative model estimation.

    Returns
    -------
    sampled_data_list : list
        The list with synthetiс datasets.
    """
    sampled_data_list = []
    for seed in seed_list:
        sampled_data = sample_stats(model, sample_len, seed)
        if scaling:
            sampled_data = scaling.inverse_transform(sampled_data)
        sampled_data_list.append(sampled_data)
    return sampled_data_list


def gmm_sample_procedure(model, sample_len, scaling, num_samples):
    """
    Sampling from GMM model.

    Parameters
    ----------
    model : instance
        Fitted generative model.

    sample_len : int
        Synthetic sample size.
        
    scaling : instance
        Fitted scaler that is applied prior to generative model estimation.
        
    num_samples : int
        Required number of synthetic datasets.

    Returns
    -------
    sampled_data_list : list
        The list with synthetiс datasets.
    """
    sampled_data_list = []
    n_samples = model.sample(sample_len * num_samples)[0]
    for i in range(num_samples):
        sampled_data = n_samples[(i * sample_len):((i + 1) * sample_len)]
        if scaling:
            sampled_data = scaling.inverse_transform(sampled_data)
        sampled_data_list.append(sampled_data)
    return sampled_data_list


def sample_sdv_procedure(model, sample_len, seed_list, scaling):
    """
    Sampling from SDV library model.

    Parameters
    ----------
    model : instance
        Fitted generative model.

    sample_len : int
        Synthetic sample size.
        
    seed_list : list
        The list of random seeds for each synthetic dataset.
        
    scaling : instance
        Fitted scaler that is applied prior to generative model estimation.

    Returns
    -------
    sampled_data_list : list
        The list with synthetiс datasets.
    """
    sampled_data_list = []
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)
        path = Path.cwd()
        if os.path.exists("/".join(str(path).split("\\")) + "/asid/automl_small/sample.csv.tmp"):
            os.remove("/".join(str(path).split("\\")) + "/asid/automl_small/sample.csv.tmp")
        sampled_data = model.sample(sample_len, output_file_path="/".join(
            str(path).split("\\")) + "/asid/automl_small/sample.csv.tmp")
        if scaling:
            sampled_data = scaling.inverse_transform(sampled_data)
        sampled_data_list.append(sampled_data)
    return sampled_data_list


def get_sampled_data(model, sample_len, seed_list, method, scaling):
    """
    Calls a sampling function.

    Parameters
    ----------
    model : instance
        Fitted generative model.

    sample_len : int
        Synthetic sample size.
        
    seed_list : list
        The list of random seeds for each synthetic dataset.
        
    method : {"sklearn_kde", "stats_kde_cv_ml", "stats_kde_cv_ls", "gmm", "bayesian_gmm", "ctgan",
        "copula", "copulagan", "tvae"}
        Generative algorithm label.
        
    scaling : instance
        Fitted scaler that is applied prior to generative model estimation.

    Returns
    -------
    sampled_data_list : list
        The list with synthetiс datasets.
    """
    if method in ["sklearn_kde"]:
        sampled_data_list = simple_sample_sklearn_procedure(model, sample_len, seed_list, scaling)
    elif method in ["stats_kde_cv_ml", "stats_kde_cv_ls"]:
        sampled_data_list = simple_sample_stats_procedure(model, sample_len, seed_list, scaling)
    elif method in ["gmm", "bayesian_gmm"]:
        sampled_data_list = gmm_sample_procedure(model, sample_len, scaling, len(seed_list))
    elif method in ["ctgan", "copula", "copulagan", "tvae"]:
        sampled_data_list = sample_sdv_procedure(model, sample_len, seed_list, scaling)
    return sampled_data_list
