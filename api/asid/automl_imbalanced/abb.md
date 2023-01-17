Module asid.automl_imbalanced.abb
=================================
This module contains AutoBalanceBoost class.

Classes
-------

`AutoBalanceBoost(num_iter=40, num_est=16)`
:   AutoBalanceBoost classifier is a tailored imbalanced learning framework
    with the built-in hyper-parameters tuning procedure.
    
    Parameters
    ----------
    num_iter : int, default=40
        The number of boosting iterations.
    
    num_est : int, default=16
        The number of estimators in the base ensemble.
    
    Attributes
    ----------
    ensemble_ : list
        The list of fitted ensembles that constitute AutoBalanceBoost model.
    
    param_ : dict
        The optimal values of AutoBalanceBoost hyper-parameters.

    ### Methods

    `feature_importances(self)`
    :   Calculates normalized feature importances.
        
        Returns
        -------
        feat_imp : array-like
            The normalized feature importances.

    `fit(self, X, y)`
    :   Fits AutoBalanceBoost model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training sample.
        
        y : array-like
            The target values.
        
        Returns
        -------
        self : AutoBalanceBoost classifier
            Fitted estimator.

    `predict(self, X)`
    :   Predicts class label.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test sample.
        
        Returns
        -------
        pred : array-like
            The predicted class.

    `predict_proba(self, X)`
    :   Predicts class probability.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test sample.
        
        Returns
        -------
        pred_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.