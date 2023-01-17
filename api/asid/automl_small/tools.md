Module asid.automl_small.tools
==============================
This module contains helping functions for GenerativeModel class.

Functions
---------

    
`check_gen_model_list(metric)`
:   

    
`check_gm_fitted(self)`
:   

    
`check_num_type(x, num_type, num_cl)`
:   

    
`check_sim_metric_list(metric, mtype)`
:   

    
`check_x_y(X, y=None)`
:   

    
`choose_and_fit_model(data, similarity_metric, scaler, data_scaled, num_syn_samples, hyp_time)`
:   Chooses an optimal generative model and fits GenerativeModel instance.
    
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