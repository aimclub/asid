"""
This module contains helping functions for AutoBalanceBoost class.
"""

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn import tree
from sklearn.metrics import f1_score


def get_newds(pred_proba, ts, X, Y, num_mod, balanced, num_feat, feat_gen, feat_imp, ts_gen):
    """
    Samples train datasets for bagging during the boosting phase.

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
    """
    train_datasets = []
    pred_proba_res = pred_proba.copy()
    pred_proba_res = 1 - pred_proba_res
    pred_proba_res[pred_proba_res == 0] = 1e-5
    class_prop = []
    np.random.seed(42)
    for i in range(num_mod):
        if isinstance(ts, list):
            if len(ts) == 0:
                ts_opt = ts[0]
            else:
                ts_opt = ts_gen.choice(ts, size=None, replace=False)
        else:
            ts_opt = ts
        sub_train_dataset = {}
        if balanced:
            class_unique, class_count = np.unique(Y, return_counts=True)
            pred_proba_res_norm = pred_proba_res / np.sum(pred_proba_res)
            pred_proba_res_balanced = pred_proba_res_norm.copy()
            if isinstance(balanced, dict):
                maj_class = class_unique[np.argmax(class_count)]
                num_classes_balanced = class_unique.shape[0] - balanced["Not_balanced"].shape[0]
                balance_share = 0
                for class_v in balanced["Not_balanced"]:
                    class_arg = np.where(class_unique == class_v)[0]
                    balance_share += class_count[class_arg] / Y.shape[0]
                target_maj_share = 1 / (1 + balanced["balance"] * (num_classes_balanced - 1) + balance_share)
                target_min_share = (1 - target_maj_share - balance_share) / (num_classes_balanced - 1)
                for n, class_val in enumerate(class_unique):
                    if class_val in balanced["Not_balanced"]:
                        target_share = class_count[n] / Y.shape[0]
                    else:
                        if class_val == maj_class:
                            target_share = target_maj_share
                        else:
                            target_share = target_min_share
                    sum_proba_class = np.sum(pred_proba_res_norm[Y == class_val])
                    diff_sum = sum_proba_class - target_share
                    pred_proba_res_balanced[Y == class_val] = pred_proba_res_balanced[Y == class_val] * (
                            1 - diff_sum / sum_proba_class)
                new_train_ind = np.random.choice(list(range(X.shape[0])), int(X.shape[0] * ts_opt), replace=True,
                                                 p=pred_proba_res_balanced)
            else:
                maj_class = class_unique[np.argmax(class_count)]
                maj_share = 1 / (1 + balanced * (class_unique.shape[0] - 1))
                for class_val in class_unique:
                    if class_val == maj_class:
                        target_share = maj_share
                    else:
                        target_share = (1 - maj_share) / (class_unique.shape[0] - 1)
                    sum_proba_class = np.sum(pred_proba_res_norm[Y == class_val])
                    diff_sum = sum_proba_class - target_share
                    pred_proba_res_balanced[Y == class_val] = pred_proba_res_balanced[Y == class_val] * (
                            1 - diff_sum / sum_proba_class)
                new_train_ind = np.random.choice(list(range(X.shape[0])), int(X.shape[0] * ts_opt), replace=True,
                                                 p=pred_proba_res_balanced)
        else:
            pred_proba_res_norm = pred_proba_res / np.sum(pred_proba_res)
            new_train_ind = np.random.choice(list(range(X.shape[0])), int(X.shape[0] * ts_opt), replace=True,
                                             p=pred_proba_res_norm)
        sub_train_dataset["X_train"] = X[new_train_ind]
        if num_feat != X.shape[1]:
            sub_train_dataset["X_train"] = choose_feat(sub_train_dataset["X_train"], num_feat, feat_gen, feat_imp)
        sub_train_dataset["Y_train"] = Y[new_train_ind]
        train_datasets.append(sub_train_dataset)
        class_prop.append(
            np.unique(sub_train_dataset["Y_train"], return_counts=True)[1] / sub_train_dataset["Y_train"].shape[0])
    return train_datasets, class_prop


def other_ensemble_procedure(X, train_datasets, pred_proba_list, model_list, classes_sorted_train):
    """
    Fits bagging during the boosting phase.

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
    """
    sub_model_list = []
    sub_pred_proba_list = []
    for element in train_datasets:
        clf = tree.DecisionTreeClassifier(random_state=42)
        clf.fit(element["X_train"], element["Y_train"])
        pred_proba = clf.predict_proba(X)
        if pred_proba.shape[1] < len(classes_sorted_train):
            classes_sorted_model = np.argsort(clf.classes_)
            classes_dict_model = dict(zip(clf.classes_, classes_sorted_model))
            res_proba = []
            for cl in classes_sorted_train:
                if cl not in classes_dict_model:
                    res_proba.append(np.array([0] * pred_proba.shape[0]))
                else:
                    res_proba.append(pred_proba[:, classes_dict_model[cl]])
            pred_proba = np.column_stack(res_proba)
        else:
            classes_sorted = np.argsort(clf.classes_)
            pred_proba = pred_proba[:, classes_sorted]
        sub_pred_proba_list.append(pred_proba.copy())
        sub_model_list.append(clf)
    pred_proba_mean = np.mean(sub_pred_proba_list, axis=0)
    pred_proba_list.append(pred_proba_mean)
    model_list.append(sub_model_list)
    return pred_proba_list, model_list


def get_bootstrap_balanced_samples(X, Y, balanced, ts, sample_gen):
    """
    Balancing procedure at the first iteration.

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
    """
    X_sampled = []
    Y_sampled = []
    class_unique, class_count = np.unique(Y, return_counts=True)
    maj_class = class_unique[np.argmax(class_count)]
    if isinstance(balanced, dict):
        y_not_balanced = Y[np.isin(Y, balanced["Not_balanced"])].shape[0]
        maj_share = 1 / (1 + balanced["balance"] * (class_unique.shape[0] - 1 - balanced["Not_balanced"].shape[0]))
        min_share = (1 - maj_share) / (class_unique.shape[0] - 1 - balanced["Not_balanced"].shape[0])
        for i, class_val in enumerate(class_unique):
            if class_val in balanced["Not_balanced"]:
                indices = sample_gen.integers(0, X[Y == class_val].shape[0], int(X[Y == class_val].shape[0] * ts))
            else:
                if class_val == maj_class:
                    indices = sample_gen.integers(0, X[Y == class_val].shape[0],
                                                  int((X.shape[0] - y_not_balanced) * maj_share * ts))
                else:
                    indices = sample_gen.integers(0, X[Y == class_val].shape[0],
                                                  int((X.shape[0] - y_not_balanced) * min_share * ts))
            X_sampled.extend(X[Y == class_val][indices])
            Y_sampled.extend([class_val for element in indices])
    elif balanced == False:
        indices = sample_gen.integers(0, len(X), int(len(X) * ts))
        X_sampled.extend(X[indices])
        Y_sampled.extend(Y[indices])
    else:
        maj_share = 1 / (1 + balanced * (class_unique.shape[0] - 1))
        for class_val in class_unique:
            if class_val == maj_class:
                indices = sample_gen.integers(0, len(X[Y == class_val]), int(len(X) * maj_share * ts))
            else:
                indices = sample_gen.integers(0, len(X[Y == class_val]),
                                              int(len(X) * (1 - maj_share) / (class_unique.shape[0] - 1) * ts))
            X_sampled.extend(X[Y == class_val][indices])
            Y_sampled.extend([class_val for element in indices])
    X_sampled = np.vstack(X_sampled)
    Y_sampled = np.hstack(Y_sampled)
    return X_sampled, Y_sampled


def choose_feat(X, n, feat_gen, feat_imp):
    """
    Samples the zeroed features.

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
    """
    feat_imp[feat_imp == 0] = 1e-7
    feat_imp = np.array(feat_imp) / sum(feat_imp)
    choose_feat = feat_gen.choice(list(range(X.shape[1])), int(X.shape[1] - n), replace=False, p=feat_imp)
    X[:, choose_feat] = 0
    return X


def first_ensemble_procedure(X, Y, ts, num_mod, balanced, num_feat, feat_gen, res_feat_imp, classes_sorted_train,
                             ts_gen):
    """
    Fits bagging at the first iteration.

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
    """
    sample_gen = np.random.default_rng(seed=42)
    sub_model_list = []
    model_list = []
    pred_proba_list = []
    sub_pred_proba_list = []
    feat_imp_list = []
    for i in range(num_mod):
        if isinstance(ts, list):
            if len(ts) == 0:
                ts_opt = ts[0]
            else:
                ts_opt = ts_gen.choice(ts, size=None, replace=False)
        else:
            ts_opt = ts
        X_sampled, Y_sampled = get_bootstrap_balanced_samples(X, Y, balanced, ts_opt, sample_gen)
        if num_feat != X.shape[1]:
            X_sampled = choose_feat(X_sampled, num_feat, feat_gen, res_feat_imp)
        clf = tree.DecisionTreeClassifier(random_state=42)
        clf.fit(X_sampled, Y_sampled)
        pred_proba = clf.predict_proba(X)
        if pred_proba.shape[1] < len(classes_sorted_train):
            classes_sorted_model = np.argsort(clf.classes_)
            classes_dict_model = dict(zip(clf.classes_, classes_sorted_model))
            res_proba = []
            for cl in classes_sorted_train:
                if cl not in classes_dict_model:
                    res_proba.append(np.array([0] * pred_proba.shape[0]))
                else:
                    res_proba.append(pred_proba[:, classes_dict_model[cl]])
            pred_proba = np.column_stack(res_proba)
        else:
            classes_sorted = np.argsort(clf.classes_)
            pred_proba = pred_proba[:, classes_sorted]
        sub_pred_proba_list.append(pred_proba.copy())
        sub_model_list.append(clf)
        feat_imp_list.append(clf.feature_importances_)
    pred_proba_mean = np.mean(sub_pred_proba_list, axis=0)
    pred_proba_list.append(pred_proba_mean)
    feat_imp_list_mean = np.mean(feat_imp_list, axis=0)
    model_list.append(sub_model_list)
    return pred_proba_list, model_list, feat_imp_list_mean


def first_ensemble_procedure_with_cv_model(X, first_model, classes_sorted_train):
    """
    Calculates the prediction probabilities of the CV bagging.

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
    """
    res_proba_list = []
    for first_model_list in first_model:
        pred_proba_list = []
        for iter in list(range(len(first_model_list))):
            sub_pred_proba_list = []
            for clf in first_model_list[iter]:
                pred_proba = clf.predict_proba(X)
                if pred_proba.shape[1] < len(classes_sorted_train):
                    classes_sorted_model = np.argsort(clf.classes_)
                    classes_dict_model = dict(zip(clf.classes_, classes_sorted_model))
                    res_proba = []
                    for cl in classes_sorted_train:
                        if cl not in classes_dict_model:
                            res_proba.append(np.array([0] * pred_proba.shape[0]))
                        else:
                            res_proba.append(pred_proba[:, classes_dict_model[cl]])
                    pred_proba = np.column_stack(res_proba)
                else:
                    classes_sorted = np.argsort(clf.classes_)
                    pred_proba = pred_proba[:, classes_sorted]
                sub_pred_proba_list.append(pred_proba.copy())
            sub_pred_proba_mean = np.mean(sub_pred_proba_list, axis=0)
            pred_proba_list.append(sub_pred_proba_mean)
        pred_proba_mean = np.mean(pred_proba_list, axis=0)
        res_proba_list.append(pred_proba_mean)
    res_proba_mean = [np.mean(res_proba_list, axis=0)]
    model_list = [first_model]
    return res_proba_mean, model_list


def fit_ensemble(X, Y, ts, iter_lim, num_mod, balanced, first_model, num_feat, feat_imp, classes_):
    """
    Iteratively fits the resulting ensemble.

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
    """
    feat_gen = np.random.default_rng(seed=42)
    ts_gen = np.random.default_rng(seed=42)
    if not first_model:
        pred_proba_list, model_list, feat_imp_list_mean = first_ensemble_procedure(X, Y, ts, num_mod, balanced,
                                                                                   num_feat, feat_gen, feat_imp,
                                                                                   classes_, ts_gen)
    else:
        pred_proba_list, model_list = first_ensemble_procedure_with_cv_model(X, first_model, classes_)
        feat_imp_list_mean = feat_imp
    pred_proba_true_class = np.array(
        list(map(lambda x, y: pred_proba_list[0][y, np.where(classes_ == x)[0][0]], Y, list(range(Y.shape[0])))))
    res_class_prop = []
    for i in range(iter_lim - 1):
        train_datasets, class_prop = get_newds(pred_proba_true_class, ts, X, Y, num_mod, balanced, num_feat, feat_gen,
                                               feat_imp, ts_gen)
        pred_proba_list, model_list = other_ensemble_procedure(X, train_datasets, pred_proba_list, model_list,
                                                               classes_)
        pred_proba_mean = np.mean(pred_proba_list, axis=0)
        pred_proba_true_class = np.array(
            list(map(lambda x, y: pred_proba_mean[y, np.where(classes_ == x)[0][0]], Y, list(range(Y.shape[0])))))
        res_class_prop.append(class_prop)
    return model_list, feat_imp_list_mean


def calc_fscore(X, Y, model_list, classes_sorted_train):
    """
    Calculates the CV test score.

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
    """
    pred_proba_list = []
    for i in range(len(model_list)):
        sub_pred_proba_list = []
        for j in range(len(model_list[i])):
            pred_proba = model_list[i][j].predict_proba(X)
            if pred_proba.shape[1] < len(classes_sorted_train):
                classes_sorted_model = np.argsort(model_list[i][j].classes_)
                classes_dict_model = dict(zip(model_list[i][j].classes_, classes_sorted_model))
                res_proba = []
                for cl in classes_sorted_train:
                    if cl not in classes_dict_model:
                        res_proba.append(np.array([0] * pred_proba.shape[0]))
                    else:
                        res_proba.append(pred_proba[:, classes_dict_model[cl]])
                pred_proba = np.column_stack(res_proba)
            else:
                classes_sorted = np.argsort(model_list[i][j].classes_)
                pred_proba = pred_proba[:, classes_sorted]
            sub_pred_proba_list.append(pred_proba.copy())
        sub_pred_proba_mean = np.mean(sub_pred_proba_list, axis=0)
        pred_proba_list.append(sub_pred_proba_mean)
    pred_proba_mean = np.mean(pred_proba_list, axis=0)
    max_class_index = np.argmax(pred_proba_mean, axis=1)
    pred_mean = list(map(lambda x: classes_sorted_train[x], max_class_index))
    fscore_val_val = f1_score(Y, pred_mean, average=None, labels=classes_sorted_train)
    fscore_val = np.mean(fscore_val_val)
    return fscore_val, fscore_val_val


def cv_balance_procedure(X, Y, split_coef, classes_):
    """
    Chooses the optimal balancing strategy.

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
    """
    bagging_ensemble_param = {}
    skf = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=42)
    feature_val = [False, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    cv_val = [[] for i in feature_val]
    cv_val_val = [[] for i in feature_val]
    res_model = [[] for i in feature_val]
    res_feat_imp = [[] for i in feature_val]
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        for i, val in enumerate(feature_val):
            model_list, feat_imp = fit_ensemble(X_train, y_train, split_coef, 5, 6, val, None, X.shape[1], None,
                                                classes_)
            bootst_res, bootst_res_val = calc_fscore(X_test, y_test, model_list, classes_)
            cv_val[i].append(bootst_res)
            cv_val_val[i].append(bootst_res_val)
            res_model[i].append(model_list)
            res_feat_imp[i].append(feat_imp)
    un_y, y_val = np.unique(Y, return_counts=True)
    num_class = un_y.shape[0]
    cv_val_mean = np.mean(cv_val, axis=1)
    if num_class > 2 and np.argmax(cv_val_mean) != 0:
        cv_val_val_mean_list = []
        for i, val in enumerate(feature_val):
            cv_val_val_mean_list.append(np.mean(cv_val_val[i], axis=0))
        feat_num = np.argmax(cv_val_mean)
        balance_shares = []
        for val in y_val:
            if val == np.max(y_val):
                balance_shares.append(1 / (1 + feature_val[feat_num] * (num_class - 1)))
            else:
                maj_share = 1 / (1 + feature_val[feat_num] * (num_class - 1))
                balance_shares.append((1 - maj_share) / (num_class - 1))
        balance_shares = np.array(balance_shares)
        fact_shares = y_val / np.sum(y_val)
        ind_sampl = (balance_shares > fact_shares) & (cv_val_val_mean_list[0] >= cv_val_val_mean_list[feat_num])
        if ind_sampl.any():
            ind_val = un_y[ind_sampl]
            feature_val.append({"Not_balanced": ind_val, "balance": feature_val[feat_num]})
            cv_val.append([])
            res_model.append([])
            res_feat_imp.append([])
            for train_index, test_index in skf.split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                model_list, feat_imp = fit_ensemble(X_train, y_train, split_coef, 5, 6,
                                                    {"Not_balanced": ind_val, "balance": feature_val[feat_num]}, None,
                                                    X.shape[1], None, classes_)
                bootst_res, bootst_res_val = calc_fscore(X_test, y_test, model_list, classes_)
                cv_val[9].append(bootst_res)
                res_model[9].append(model_list)
                res_feat_imp[9].append(feat_imp)
            cv_val_mean = np.mean(cv_val, axis=1)
    bagging_ensemble_param["opt_balance"] = feature_val[np.argmax(cv_val_mean)]
    bagging_ensemble_param["cv_result_balance"] = cv_val[np.argmax(cv_val_mean)]
    bagging_ensemble_param["cv_balance"] = cv_val_mean
    bagging_ensemble_param["split_coef_balance"] = split_coef
    bagging_ensemble_param["models_balance"] = res_model[np.argmax(cv_val_mean)]
    bagging_ensemble_param["feat_imp_balance"] = res_feat_imp[np.argmax(cv_val_mean)]
    return bagging_ensemble_param


def calc_share(series_a, series_b, sample_gen1, sample_gen2):
    """
    Calculates performance shares for different bagging share values.

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
    """
    count = 0
    for i in range(1000):
        indices1 = sample_gen1.integers(0, len(series_a), len(series_a))
        indices2 = sample_gen2.integers(0, len(series_b), len(series_b))
        sub_series_a = np.mean(series_a[indices1])
        sub_series_b = np.mean(series_b[indices2])
        if sub_series_a >= sub_series_b:
            count += 1
    share = round(count / 1000, 3)
    return share


def get_best_bc(split_range, f_score_list, sample_gen1, sample_gen2):
    """
    Chooses a list of bagging shares with the best performance.

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
    """
    split_sorted = np.argsort(np.mean(f_score_list, axis=1))[::-1]
    split_k = np.array(f_score_list[split_sorted[0]])
    split_arg = []
    ind_bc = []
    for m in range(len(split_sorted)):
        split_m = np.array(f_score_list[split_sorted[m]])
        share = calc_share(split_m, split_k, sample_gen1, sample_gen2)
        if share > 0:
            split_arg.append(split_range[split_sorted[m]])
            ind_bc.append(split_sorted[m])
    return split_arg, ind_bc


def cv_split_procedure(X, Y, bagging_ensemble_param):
    """
    Chooses an optimal list of bagging shares.

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
    """
    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    feature_val = np.arange(0.1, 1.15, 0.05)
    cv_val_seq = [[] for i in feature_val]
    count = 0
    feat_imp_list = []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        for i, val in enumerate(feature_val):
            model_list, feat_imp = fit_ensemble(X_train, y_train, val, 5, 6, bagging_ensemble_param["opt_balance"],
                                                None, X.shape[1], None, bagging_ensemble_param["classes"])
            feat_imp_list.append(feat_imp)
            sample_gen = np.random.default_rng(seed=42)
            for k in range(100):
                indices = sample_gen.integers(0, X_test.shape[0], X_test.shape[0])
                bootst_res, bootst_res_val = calc_fscore(X_test[indices], y_test[indices], model_list,
                                                         bagging_ensemble_param["classes"])
                cv_val_seq[i].append(bootst_res)
        count += 1
    sample_gen1 = np.random.default_rng(seed=42)
    sample_gen2 = np.random.default_rng(seed=45)
    bc_seq, ind_bc = get_best_bc(feature_val, cv_val_seq, sample_gen1, sample_gen2)
    bagging_ensemble_param["opt_split"] = bc_seq
    bagging_ensemble_param["split_range"] = feature_val
    bagging_ensemble_param["feat_imp_norm"] = np.mean(np.array(feat_imp_list)[ind_bc], axis=0)
    return bagging_ensemble_param


def num_feat_procedure(X, Y, bagging_ensemble_param):
    """
    Chooses an optimal number of zeroed features.

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
    """
    skf = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=42)
    feat_imp_norm = bagging_ensemble_param["feat_imp_norm"]
    feat_imp_norm = feat_imp_norm / sum(feat_imp_norm)
    feat_imp_norm_sort_ind = np.argsort(feat_imp_norm)[::-1]
    feat_imp_norm_cumsum = np.cumsum(feat_imp_norm[feat_imp_norm_sort_ind])
    ind_chosen = feat_imp_norm_sort_ind[feat_imp_norm_cumsum < 0.95]
    feature_val = [X.shape[1] - element for element in list(range(ind_chosen.shape[0]))]
    res_model = [[] for i in feature_val]
    cv_val = [[] for i in feature_val]
    count = 0
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        for i, val in enumerate(feature_val):
            model_list, feat_imp_list_mean = fit_ensemble(X_train, y_train, bagging_ensemble_param["opt_split"], 5, 6,
                                                          bagging_ensemble_param["opt_balance"], None, val,
                                                          feat_imp_norm, bagging_ensemble_param["classes"])
            res_model[i].append(model_list)
            bootst_res, bootst_res_val = calc_fscore(X_test, y_test, model_list, bagging_ensemble_param["classes"])
            cv_val[i].append(bootst_res)
        count += 1
    cv_val_mean = np.mean(cv_val, axis=1)
    bagging_ensemble_param["opt_feat"] = feature_val[np.argmax(cv_val_mean)]
    res_model = res_model[np.argmax(cv_val_mean)]
    return bagging_ensemble_param, res_model


def boosting_of_bagging_procedure(X_train, y_train, num_iter, num_mod):
    """
    Fits an AutoBalanceBoost model.

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
    """
    boosting_params = {}
    y_val, y_counts = np.unique(y_train, return_counts=True)
    if y_counts.min() / y_counts.max() >= 0.9:
        balance_param = {"opt_balance": False, "split_coef_balance": 0}
    else:
        balance_param = cv_balance_procedure(X_train, y_train, 0.3, y_val)
    balance_param["classes"] = y_val
    boosting_params["balance_share"] = balance_param["opt_balance"]
    balance_param = cv_split_procedure(X_train, y_train, balance_param)
    boosting_params["bagging_share"] = balance_param["opt_split"]
    balance_param, first_model = num_feat_procedure(X_train, y_train, balance_param)
    boosting_params["features_number"] = balance_param["opt_feat"]
    model_list, feat_imp_list_mean = fit_ensemble(X_train, y_train, boosting_params["bagging_share"], num_iter, num_mod,
                                                  boosting_params["balance_share"], first_model,
                                                  boosting_params["features_number"], balance_param["feat_imp_norm"],
                                                  balance_param["classes"])
    return model_list, boosting_params


def get_pred(model_list, X_test):
    """
    Predicts class labels.

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
    """
    pred_list = []
    for i in range(len(model_list)):
        for j in range(len(model_list[i])):
            if not isinstance(model_list[i][j], list):
                pred = model_list[i][j].predict(X_test)
                pred_list.append(pred)
            else:
                for cv_i in range(len(model_list[i][j])):
                    for iter in range(len(model_list[i][j][cv_i])):
                        pred = model_list[i][j][cv_i][iter].predict(X_test)
                        pred_list.append(pred)
    pred_list = np.array(pred_list)
    pred_mean_hard = np.array(list(
        map(lambda y: np.unique(pred_list[:, y])[np.argmax(np.unique(pred_list[:, y], return_counts=True)[1])],
            list(range(X_test.shape[0])))))
    return pred_mean_hard


def get_pred_proba(model_list, X_test):
    """
    Predicts class probabilities.

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
    """
    pred_list = []
    for i in range(len(model_list)):
        for j in range(len(model_list[i])):
            if not isinstance(model_list[i][j], list):
                pred = model_list[i][j].predict(X_test)
                pred_list.append(pred)
            else:
                for cv_i in range(len(model_list[i][j])):
                    for iter in range(len(model_list[i][j][cv_i])):
                        pred = model_list[i][j][cv_i][iter].predict(X_test)
                        pred_list.append(pred)
    pred_list = np.array(pred_list)
    proba_mean_hard = []
    col_names = np.unique(pred_list)
    for cl_val in col_names:
        proba_mean_hard.append(np.array(list(
            map(lambda y: (pred_list[:, y] == cl_val).sum() / pred_list[:, y].shape[0], list(range(X_test.shape[0]))))))
    proba_mean_hard = np.vstack(proba_mean_hard).T
    return proba_mean_hard


def get_feat_imp(model_list):
    """
    Returns normalized feature importances.

    Parameters
    ----------
    model_list : list
        Fitted base estimators in AutoBalanceBoost.

    Returns
    -------
    feat_imp_norm : array-like
        Normalized feature importances.
    """
    feat_imp_list = []
    for i in range(len(model_list)):
        for j in range(len(model_list[i])):
            if not isinstance(model_list[i][j], list):
                feat_imp = model_list[i][j].feature_importances_
                feat_imp_list.append(feat_imp)
            else:
                for cv_i in range(len(model_list[i][j])):
                    for iter_ in range(len(model_list[i][j][cv_i])):
                        feat_imp = model_list[i][j][cv_i][iter_].feature_importances_
                        feat_imp_list.append(feat_imp)

    feat_imp_list = np.mean(feat_imp_list, axis=0)
    feat_imp_norm = feat_imp_list / np.sum(feat_imp_list)
    return feat_imp_norm
