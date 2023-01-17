Module asid.automl_small.generative_model_estimation
====================================================
This module contains functions for generative models estimation.

Functions
---------

    
`calc_bayesian_gmm_acc(params, data)`
:   Estimates the performance of BayesianGaussianMixture model with hyper-parameters values.
    
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

    
`calc_gmm_acc(params, data)`
:   Estimates the performance of GaussianMixture model with hyper-parameters values.
    
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

    
`calc_kde_acc(params, data)`
:   Estimates the performance of KDE (sklearn implementation) with hyper-parameters values.
    
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

    
`calc_sdv_acc(params, data, alg)`
:   Estimates the performance of SDV model with hyper-parameters values.
    
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

    
`fit_model(gen_algorithm, data, hyp_time)`
:   Fits generative model.
    
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

    
`get_bayesian_gmm_model(data, hyp_time)`
:   Estimates Bayesian GMM model.
    
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

    
`get_copula_model(data)`
:   Estimates Copula model.
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.
    
    Returns
    -------
    model : instance
        Fitted generative model.

    
`get_copulagan_model(data, hyp_time)`
:   Estimates CopulaGAN model.
    
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

    
`get_ctgan_model(data, hyp_time)`
:   Estimates CTGAN model.
    
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

    
`get_gmm_model(data, hyp_time)`
:   Estimates GMM model.
    
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

    
`get_tvae_model(data, hyp_time)`
:   Estimates TVAE model
    
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

    
`sklearn_kde(data, hyp_time)`
:   Estimates KDE (sklearn implementation).
    
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

    
`stats_kde(data, method)`
:   Estimates KDE (Statsmodels).
    
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