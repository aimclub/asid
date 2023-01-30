from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from numpy import ndarray
from typing import Union, Tuple


def ks_test(df1: ndarray, df2: ndarray) -> Tuple[list, list]:
    """
    Kolmogorov-Smirnov test applied to each marginal distribution.

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
    """
    p_val_list = []
    stat_list = []
    for element in list(range(df1.shape[1])):
        res = stats.ks_2samp(df1[:, element], df2[:, element])
        p_val_list.append(res[1])
        stat_list.append(res[0])
    return p_val_list, stat_list


def c2st_roc_auc(df1: ndarray, df2: ndarray) -> float:
    """
    Classifier Two-Sample Test: ROC AUC for gradient boosting classifier.

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
    """
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    df1["Class"] = 1
    df2["Class"] = 0
    data = pd.concat([df1, df2], axis=0)
    y = data['Class']
    data.drop('Class', axis=1, inplace=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
    clf = xgb.XGBClassifier(seed=10)
    score = []
    for train_index, test_index in skf.split(data, y):
        x0, x1 = data.iloc[train_index], data.iloc[test_index]
        y0, y1 = y.iloc[train_index], y.iloc[test_index]
        clf.fit(x0, y0, eval_set=[(x1, y1)], eval_metric='logloss', verbose=False, early_stopping_rounds=10)
        prval = clf.predict_proba(x1)[:, 1]
        ras = roc_auc_score(y1, prval)
        score.append(ras)
    roc_auc = np.mean(score)
    return roc_auc


def c2st_accuracy(data_orig: ndarray, sampled: ndarray) -> Tuple[float, float]:
    """
    Classifier Two-Sample Test: LOO Accuracy for 1-NN classifier.

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
    """
    data_orig = pd.DataFrame(data_orig.copy())
    sampled = pd.DataFrame(sampled.copy())
    data_orig["class"] = 1
    sampled["class"] = 0
    data_res = pd.concat([data_orig, sampled], ignore_index=True, sort=False)
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    n_var = data_res.shape[1] - 1
    for train_index, test_index in loo.split(data_res):
        x_train, x_test = data_res.iloc[train_index, :n_var - 1], data_res.iloc[test_index, :n_var - 1]
        y_train, y_test = data_res.iloc[train_index, n_var], data_res.iloc[test_index, n_var]
        nn_clf = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
        preds = nn_clf.predict(x_test)
        y_true.extend(y_test)
        y_pred.extend(preds)
    res = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
    res_1 = res[res["y_true"] == 1]
    acc_r = res_1["y_pred"].sum() / len(res_1)
    res_0 = res[res["y_true"] == 0]
    acc_g = 1 - res_0["y_pred"].sum() / len(res_0)
    return acc_r, acc_g


def ks_permutation(stat: list, df1: ndarray, df2: ndarray) -> float:
    """
    Kolmogorov-Smirnov permutation test applied to each maginal distribution.

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
    """
    p_val = None
    i = 0
    p_val_list = []
    while p_val is None:
        x1 = df1[:, i]
        x2 = df2[:, i]
        p_val_stat = ks_permutation_var(stat[i], x1, x2)
        if p_val_stat < 0.05:
            p_val = p_val_stat
        else:
            i += 1
            p_val_list.append(p_val_stat)
        if i == df1.shape[1] - 1:
            p_val = p_val_list[0]
    return p_val


def ks_permutation_var(stat: float, series1: ndarray, series2: ndarray) -> float:
    """
    Kolmogorov-Smirnov permutation test for marginal distribution.

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
    """
    x1 = series1
    x2 = series2
    lx1 = len(x1)
    lx2 = len(x2)
    data_x = np.concatenate([x1, x2], axis=0)
    rng = np.random.default_rng(seed=42)
    ks_res = []
    n_samp = 1000
    for j in range(n_samp):
        x_con = rng.permutation(data_x)
        x1_perm = x_con[:lx1]
        x2_perm = x_con[lx2:]
        ks_res.append(stats.ks_2samp(x1_perm, x2_perm)[0])
    ks_list = np.sort(ks_res)
    ks_arg = np.arange(start=1, stop=n_samp + 1) / n_samp
    p_val = 1 - np.interp(stat, ks_list, ks_arg)
    return p_val


def zu_overfitting_statistic(df1: ndarray, df2: ndarray, df3: ndarray) -> float:
    """
    Zu overfitting statistic calculation.

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
    """
    m = df2.shape[0]
    n = df1.shape[0]
    t_nn = NearestNeighbors(n_neighbors=1).fit(df3)
    lqm, _ = t_nn.kneighbors(X=df2, n_neighbors=1)
    lpn, _ = t_nn.kneighbors(X=df1, n_neighbors=1)
    u, p = mannwhitneyu(lqm, lpn, alternative='less')
    mean = (n * m / 2) - 0.5
    std = np.sqrt(n * m * (n + m + 1) / 12)
    zu_stat = (u - mean) / std
    return zu_stat


def calc_metrics(data: ndarray, sampled_data: ndarray, metric: str, test_data: Union[None, ndarray] = None) -> \
        Union[float, list]:
    """
    Calculates dataset similarity metrics.

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
    """
    if metric == "c2st_acc":
        acc_r, acc_g = c2st_accuracy(data, sampled_data)
        result = np.mean([acc_r, acc_g])
    elif metric == "zu":
        result = zu_overfitting_statistic(test_data, sampled_data, data)
    elif metric == "roc_auc":
        result = c2st_roc_auc(data, sampled_data)
    elif metric == "ks_test":
        ks_p_val_list, ks_stat_list = ks_test(data, sampled_data)
        ks_p_val = ks_permutation(ks_stat_list, data, sampled_data)
        n = np.argmax(ks_stat_list)
        stat = ks_stat_list[n]
        result = [stat, ks_p_val]
    return result
