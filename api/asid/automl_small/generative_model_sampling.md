Module asid.automl_small.generative_model_sampling
==================================================
This module contains functions for sampling from fitted generative models.

Functions
---------

    
`get_sampled_data(model, sample_len, seed_list, method, scaling)`
:   Calls a sampling function.
    
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

    
`gmm_sample_procedure(model, sample_len, scaling, num_samples)`
:   Sampling from GMM model.
    
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

    
`sample_sdv_procedure(model, sample_len, seed_list, scaling)`
:   Sampling from SDV library model.
    
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

    
`sample_stats(kde, size, seed)`
:   Base sampling procedure from Statsmodel's KDE.
    
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

    
`simple_sample_sklearn_procedure(model, sample_len, seed_list, scaling)`
:   Sampling synthetic datasets from sklearn KDE.
    
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

    
`simple_sample_stats_procedure(model, sample_len, seed_list, scaling)`
:   Sampling synthetic datasets from Statsmodel's KDE.
    
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