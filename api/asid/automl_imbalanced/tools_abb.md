Module asid.automl_imbalanced.tools_abb
=======================================
This module contains helping functions for AutoBalanceBoost class.

Functions
---------

    
`boosting_of_bagging_procedure(X_train, y_train, num_iter, num_mod)`
:   Fits an AutoBalanceBoost model.
    
    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training sample.
    
    y_train : array-like
        The target values.
    
    num_iter : int
        The number of boosting iterations.
    
    num_mod : int
        The number of estimators in the base ensemble.
    
    Returns
    -------
    model_list : list
        Fitted base estimators in AutoBalanceBoost.
    
    boosting_params : dict
        CV procedure data.

    
`calc_fscore(X, Y, model_list, classes_sorted_train)`
:   Calculates the CV test score.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    Y : array-like
        The target values.
    
    model_list : list
        Fitted base estimators in AutoBalanceBoost.
    
    classes_sorted_train : array-like
        Class labels.
    
    Returns
    -------
    fscore_val : float
        CV test score.
    
    fscore_val_val : array-like
        CV test score for each class separately.

    
`calc_share(series_a, series_b, sample_gen1, sample_gen2)`
:   Calculates performance shares for different bagging share values.
    
    Parameters
    ----------
    series_a : array-like
        Scores for bagging share value for a range of splitting iterations.
    
    series_b : array-like
        Scores for bagging share value with the highest mean score for a range of splitting iterations.
    
    sample_gen1 : instance
        Random sample generator.
    
    sample_gen2 : instance
        Random sample generator.
    
    Returns
    -------
    share : float
        Performance share for bagging share value.

    
`choose_feat(X, n, feat_gen, feat_imp)`
:   Samples the zeroed features.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    n : int
        The number of features that are not zeroed.
    
    feat_gen : instance
        Random sample generator.
    
    feat_imp : array-like
        Normalized feature importances.
    
    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        Training sample with zeroed features.

    
`cv_balance_procedure(X, Y, split_coef, classes_)`
:   Chooses the optimal balancing strategy.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    Y : array-like
        The target values.
    
    split_coef : float
        Train sample share for base learner estimation.
    
    classes_ : array-like
        Class labels.
    
    Returns
    -------
    bagging_ensemble_param : dict
        CV procedure data.

    
`cv_split_procedure(X, Y, bagging_ensemble_param)`
:   Chooses an optimal list of bagging shares.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    Y : array-like
        The target values.
    
    bagging_ensemble_param : dict
        CV procedure data.
    
    Returns
    -------
    bagging_ensemble_param : dict
        CV procedure data.

    
`first_ensemble_procedure(X, Y, ts, num_mod, balanced, num_feat, feat_gen, res_feat_imp, classes_sorted_train, ts_gen)`
:   Fits bagging at the first iteration.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    Y : array-like
        The target values.
    
    ts : list
        A range of train sample shares for base learner estimation.
    
    num_mod : int
        The number of estimators in the base ensemble.
    
    balanced : bool or dict
        Balancing strategy parameter.
    
    num_feat : int
        The number of features that are not zeroed.
    
    feat_gen : instance
        Random sample generator.
    
    res_feat_imp : array-like
        Normalized feature importances.
    
    classes_sorted_train : array-like
        The sorted unique class values.
    
    ts_gen : instance
        Random sample generator.
    
    Returns
    -------
    pred_proba_list : list
        Class probabilities predicted by each base estimator in AutoBalanceBoost.
    
    model_list : list
        Fitted base estimators in AutoBalanceBoost.
    
    feat_imp_list_mean : array-like
        Normalized feature importances.

    
`first_ensemble_procedure_with_cv_model(X, first_model, classes_sorted_train)`
:   Calculates the prediction probabilities of the CV bagging.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    first_model : list
        Fitted base estimators in AutoBalanceBoost at the first iteration.
    
    classes_sorted_train : array-like
        The sorted unique class values.
    
    Returns
    -------
    res_proba_mean : list
        Class probabilities predicted at the first iteration.
    
    model_list : list
        Fitted base estimators in AutoBalanceBoost.

    
`fit_ensemble(X, Y, ts, iter_lim, num_mod, balanced, first_model, num_feat, feat_imp, classes_)`
:   Iteratively fits the resulting ensemble.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    Y : array-like
        The target values.
    
    ts : float or list
        A range of train sample shares for base learner estimation.
    
    iter_lim : int
        The number of boosting iterations.
    
    num_mod : int
        The number of estimators in the base ensemble.
    
    balanced : bool or dict
        Balancing strategy parameter.
    
    first_model : list or None
        Fitted base estimators in AutoBalanceBoost at the first iteration.
    
    num_feat : int
        The number of features that are not zeroed.
    
    feat_imp : array-like
        Normalized feature importances.
    
    classes_ : array-like
        Class labels.
    
    Returns
    -------
    model_list : list
        Fitted base estimators in AutoBalanceBoost.
    
    feat_imp_list_mean : array-like
        Normalized feature importances.

    
`get_best_bc(split_range, f_score_list, sample_gen1, sample_gen2)`
:   Chooses a list of bagging shares with the best performance.
    
    Parameters
    ----------
    split_range : array-like
        Bagging share values.
    
    f_score_list : list
        CV scores for bagging share values.
    
    sample_gen1 : instance
        Random sample generator.
    
    sample_gen2 : instance
        Random sample generator.
    
    Returns
    -------
    split_arg : list
        List of optimal bagging share values.
    
    ind_bc : list
        Indices of optimal bagging share values.

    
`get_bootstrap_balanced_samples(X, Y, balanced, ts, sample_gen)`
:   Balancing procedure at the first iteration.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    Y : array-like
        The target values.
    
    ts : list
        A range of train sample shares for base learner estimation.
    
    balanced : bool or dict
        Balancing strategy parameter.
    
    sample_gen : instance
        Random sample generator.
    
    Returns
    -------
    X_sampled : array-like of shape (n_samples, n_features)
        Generated training sample.
    
    Y_sampled : array-like
        Generated target values.

    
`get_feat_imp(model_list)`
:   Returns normalized feature importances.
    
    Parameters
    ----------
    model_list : list
        Fitted base estimators in AutoBalanceBoost.
    
    Returns
    -------
    feat_imp_norm : array-like
        Normalized feature importances.

    
`get_newds(pred_proba, ts, X, Y, num_mod, balanced, num_feat, feat_gen, feat_imp, ts_gen)`
:   Samples train datasets for bagging during the boosting phase.
    
    Parameters
    ----------
    pred_proba : array-like
        Class probabilities predicted by AutoBalanceBoost for the correct class.
    
    ts : list
        A range of train sample shares for base learner estimation.
    
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    Y : array-like
        The target values.
    
    num_mod : int
        The number of estimators in the base ensemble.
    
    balanced : bool or dict
        Balancing strategy parameter.
    
    num_feat : int
        The number of features that are not zeroed.
    
    feat_gen : instance
        Random sample generator.
    
    feat_imp : array-like
        Normalized feature importances.
    
    ts_gen : instance
        Random sample generator.
    
    Returns
    -------
    train_datasets : list
        Randomly generated train datasets for bagging.
    
    class_prop : list
        Class shares for each train dataset.

    
`get_pred(model_list, X_test)`
:   Predicts class labels.
    
    Parameters
    ----------
    model_list : list
        Fitted base estimators in AutoBalanceBoost.
    
    X_test : array-like of shape (n_samples, n_features)
        Test sample.
    
    Returns
    -------
    pred_mean_hard : array-like
        The predicted class.

    
`get_pred_proba(model_list, X_test)`
:   Predicts class probabilities.
    
    Parameters
    ----------
    model_list : list
        Fitted base estimators in AutoBalanceBoost.
    
    X_test : array-like of shape (n_samples, n_features)
        Test sample.
    
    Returns
    -------
    proba_mean_hard : array-like of shape (n_samples, n_classes)
        The predicted class probabilities.

    
`num_feat_procedure(X, Y, bagging_ensemble_param)`
:   Chooses an optimal number of zeroed features.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    Y : array-like
        The target values.
    
    bagging_ensemble_param : dict
        CV procedure data.
    
    Returns
    -------
    bagging_ensemble_param : dict
        CV procedure data.
    
    res_model : list
        Fitted base estimators in AutoBalanceBoost.

    
`other_ensemble_procedure(X, train_datasets, pred_proba_list, model_list, classes_sorted_train)`
:   Fits bagging during the boosting phase.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training sample.
    
    train_datasets : list
        Randomly generated train datasets for bagging.
    
    pred_proba_list : list
        Class probabilities predicted by each base estimator in AutoBalanceBoost.
    
    model_list : list
        Fitted base estimators in AutoBalanceBoost.
    
    classes_sorted_train : array-like
        The sorted unique class values.
    
    Returns
    -------
    pred_proba_list : list
        Class probabilities predicted by each base estimator in AutoBalanceBoost.
    
    model_list : list
        Fitted base estimators in AutoBalanceBoost.