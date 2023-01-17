Module asid.automl_small.dataset_similarity_metrics
===================================================
This module contains functions for dataset similarity metrics calculation.

Functions
---------

    
`c2st_accuracy(data_orig, sampled)`
:   Classifier Two-Sample Test: LOO Accuracy for 1-NN classifier.
    
    Parameters
    ----------
    data_orig : array-like of shape (n_samples, n_features)
        Train sample.
    
    sampled : array-like of shape (n_samples, n_features)
        Synthetic sample.
    
    Returns
    -------
    acc_r : float
        Accuracy for real samples.
    
    acc_g : float
        Accuracy for generated samples.
    
    References
    ----------
    Xu, Q. et al. (2018) “An empirical study on evaluation metrics of generative adversarial networks” arXiv preprint
    arXiv:1806.07755.

    
`c2st_roc_auc(df1, df2)`
:   Classifier Two-Sample Test: ROC AUC for gradient boosting classifier.
    
    Parameters
    ----------
    df1 : array-like of shape (n_samples, n_features)
        Train sample.
    
    df2 : array-like of shape (n_samples, n_features)
        Synthetic sample.
    
    Returns
    -------
    roc_auc : float
        ROC AUC value.
    
    References
    ----------
    Friedman, J. H. (2003) “On Multivariate Goodness–of–Fit and Two–Sample Testing” Statistical Problems in Particle
    Physics, Astrophysics and Cosmology, PHYSTAT2003: 311-313.

    
`calc_metrics(data, sampled_data, metric, test_data=None)`
:   Calculates dataset similarity metrics.
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training sample.
    
    sampled_data : array-like of shape (n_samples, n_features)
        Synthetic sample.
    
    metric : {"zu", "c2st_acc", "roc_auc", "ks_test"}
        Metric that is used to choose the optimal generative model.
    
    test_data : array-like of shape (n_samples, n_features)
        Test sample.
    
    Returns
    -------
    result : float or list
        Metric value. For "ks_test" a list [statistic, p-value] is output.

    
`ks_permutation(stat, df1, df2)`
:   Kolmogorov-Smirnov permutation test applied to each maginal distribution.
    
    Parameters
    ----------
    stat : list
        List of statistic values for marginal distributions.
    
    df1 : array-like of shape (n_samples, n_features)
        Train sample.
    
    df2 : array-like of shape (n_samples, n_features)
        Synthetic sample.
    
    Returns
    -------
    p_val : float
        P-value obtained using permutation test.

    
`ks_permutation_var(stat, series1, series2)`
:   Kolmogorov-Smirnov permutation test for marginal distribution.
    
    Parameters
    ----------
    stat : float
        Statistic value for marginal distribution.
    
    series1 : array-like
        Train sample series.
    
    series2 : array-like
        Synthetic sample series.
    
    Returns
    -------
    p_val : float
        P-value.

    
`ks_test(df1, df2)`
:   Kolmogorov-Smirnov test applied to each marginal distribution.
    
    Parameters
    ----------
    df1 : array-like of shape (n_samples, n_features)
        Train sample.
    
    df2 : array-like of shape (n_samples, n_features)
        Synthetic sample.
    
    Returns
    -------
    p_val_list : list
        List of p-values for marginal distributions.
    
    stat_list : list
        List of statistic values for marginal distributions.

    
`zu_overfitting_statistic(df1, df2, df3)`
:   Zu overfitting statistic calculation.
    
    Parameters
    ----------
    df1 : array-like of shape (n_samples, n_features)
        Test sample.
    
    df2 : array-like of shape (n_samples, n_features)
        Synthetic sample.
    
    df3 : array-like of shape (n_samples, n_features)
        Train sample.
    
    Returns
    -------
    zu_stat : float
        Metric value.
    
    References
    ----------
    Meehan C., Chaudhuri K., Dasgupta S. (2020) “A non-parametric test to detect data-copying in generative models”
    International Conference on Artificial Intelligence and Statistics.