Module asid.automl_imbalanced.ilc
=================================
This module contains ImbalancedLearningClassifier class.

Classes
-------

`ImbalancedLearningClassifier(split_num=5, hyperopt_time=0, eval_metric='f1_macro')`
:   ImbalancedLearningClassifier finds an optimal classifier among the combinations of balancing
    procedures from imbalanced-learn library (with Hyperopt optimization) and state-of-the-art ensemble classifiers,
    and the tailored classifier AutoBalanceBoost.
    
    Parameters
    ----------
    split_num : int, default=5
        The number of splitting iterations for obtaining an out-of-fold score. If the number is a 5-fold, then
        StratifiedKFold with 5 splits is repeated with the required number of seeds, otherwise StratifiedShuffleSplit
        with split_num splits is used.
    
    hyperopt_time : int, default=0
        The runtime setting (in seconds) for Hyperopt optimization. Hyperopt is used to find the optimal
        hyper-parameters for balancing procedures.
    
    eval_metric : {"accuracy", "roc_auc", "log_loss", "f1_macro", "f1_micro", "f1_weighted"}, default="f1_macro"
        Metric that is used to evaluate the model performance and to choose the best option.
    
    Attributes
    ----------
    classifer_ : instance
        Optimal fitted classifier.
    
    classifer_label_ : str
        Optimal classifier label.
    
    score_ : float
        Averaged out-of-fold value of eval_metric for the optimal classifier.
    
    scaler_ : instance
        Fitted scaler that is applied prior to classifier estimation.
    
    encoder_ : instance
        Fitted label encoder.
    
    classes_ : array-like
        Class labels.
    
    evaluated_models_scores_ : dict
        Score series for the range of estimated classifiers.
    
    evaluated_models_time_ : dict
        Time data for the range of estimated classifiers.
    
    conf_int_ : tuple
        95% confidence interval for the out-of-fold value of eval_metric for the optimal classifier.

    ### Methods

    `fit(self, X, y)`
    :   Fits ImbalancedLearningClassifier model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training sample.
        
        y : array-like
            The target values.
        
        Returns
        -------
        self : ImbalancedLearningClassifier instance
            Fitted estimator.

    `leaderboard(self)`
    :   Calculates the leaderboard statistics.
        
        Returns
        -------
        ls : dict
            The leaderboard statistics that includes sorted lists in accordance with the following indicators:
            "Mean score", "Mean rank", "Share of experiments with the first place, %",
            "Average difference with the leader, %".

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
    :   Predicts class label probability.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test sample.
        
        Returns
        -------
        pred_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.