import numpy as np
import pytest
from scipy.stats import ks_2samp
from asid.automl_small.dataset_similarity_metrics import ks_test, c2st_roc_auc, c2st_accuracy, ks_permutation, \
    ks_permutation_var, zu_overfitting_statistic, calc_metrics


# test data
@pytest.fixture
def generate_test_data():
    np.random.seed(42)
    df1 = np.random.normal(loc=0, scale=1, size=(100, 10))
    df2 = np.random.normal(loc=1, scale=1, size=(100, 10))
    df3 = np.random.normal(loc=0.5, scale=1, size=(100, 10))
    return df1, df2, df3


def test_ks_test(generate_test_data):
    df1, df2, _ = generate_test_data
    p_vals, stats = ks_test(df1, df2)
    assert len(p_vals) == df1.shape[1]
    assert len(stats) == df1.shape[1]
    assert all(0 <= p <= 1 for p in p_vals)
    assert all(0 <= s <= 1 for s in stats)


def test_c2st_roc_auc(generate_test_data):
    df1, df2, _ = generate_test_data
    roc_auc = c2st_roc_auc(df1, df2)
    assert 0 <= roc_auc <= 1


def test_c2st_accuracy(generate_test_data):
    df1, df2, _ = generate_test_data
    acc_r, acc_g = c2st_accuracy(df1, df2)
    assert 0 <= acc_r <= 1
    assert 0 <= acc_g <= 1


def test_ks_permutation(generate_test_data):
    df1, df2, _ = generate_test_data
    stat_list, _ = ks_test(df1, df2)
    p_val = ks_permutation(stat_list, df1, df2)
    assert 0 <= p_val <= 1


def test_ks_permutation_var(generate_test_data):
    df1, df2, _ = generate_test_data
    stat = ks_2samp(df1[:, 0], df2[:, 0]).statistic
    p_val = ks_permutation_var(stat, df1[:, 0], df2[:, 0])
    assert 0 <= p_val <= 1


def test_zu_overfitting_statistic(generate_test_data):
    df1, df2, df3 = generate_test_data
    zu_stat = zu_overfitting_statistic(df1, df2, df3)
    assert isinstance(zu_stat, float)


def test_calc_metrics_c2st_acc(generate_test_data):
    df1, df2, _ = generate_test_data
    result = calc_metrics(df1, df2, metric="c2st_acc")
    assert 0 <= result <= 1


def test_calc_metrics_zu(generate_test_data):
    df1, df2, df3 = generate_test_data
    result = calc_metrics(df1, df2, metric="zu", test_data=df3)
    assert isinstance(result, float)


def test_calc_metrics_roc_auc(generate_test_data):
    df1, df2, _ = generate_test_data
    result = calc_metrics(df1, df2, metric="roc_auc")
    assert 0 <= result <= 1


def test_calc_metrics_ks_test(generate_test_data):
    df1, df2, _ = generate_test_data
    result = calc_metrics(df1, df2, metric="ks_test")
    assert isinstance(result, list)
    assert len(result) == 2
    assert 0 <= result[0] <= 1
    assert 0 <= result[1] <= 1
