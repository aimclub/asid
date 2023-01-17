Module asid.automl_small.gm
===========================
This module contains GenerativeModel class.

Classes
-------

`GenerativeModel(gen_model_type='optimize', similarity_metric='zu', num_syn_samples=100, hyperopt_time=0)`
:   GenerativeModel is a tool designed to find an appropriate generative model for small tabular data. It estimates the
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

    ### Methods

    `fit(self, data)`
    :   Fits GenerativeModel instance.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Training sample.
        
        Returns
        -------
        self : GenerativeModel instance
            Fitted generative model.

    `sample(self, sample_size, random_state=42)`
    :   Generates synthetic sample from GenerativeModel.
        
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

    `score(self, train_data, similarity_metric='zu', test_data=None)`
    :   Evaluates the similarity of GenerativeModel samples and train data with the specified similarity metric.
        
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