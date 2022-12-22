from asid.automl_small.gm import GenerativeModel
from sklearn.datasets import load_iris


def test_gm_opt():
    X = load_iris().data
    clf = GenerativeModel(gen_model_type="optimize", num_syn_samples=10, hyperopt_time=5, similarity_metric="zu")
    clf.fit(X)
    assert clf.gen_model_ is not None
    assert clf.scaler_ is not None
    assert len(clf.info_) != 0
    assert isinstance(clf.score_, float)
    assert isinstance(clf.gen_model_label_, str)


def test_gm_sample():
    X = load_iris().data
    clf = GenerativeModel(gen_model_type="sklearn_kde", num_syn_samples=1, hyperopt_time=3, similarity_metric="zu")
    clf.fit(X)
    ds = clf.sample(1000)
    assert ds.shape[0] == 1000
    assert ds.shape[1] == X.shape[1]


def test_gm_score():
    X = load_iris().data
    clf = GenerativeModel(gen_model_type="sklearn_kde", num_syn_samples=1, hyperopt_time=3, similarity_metric="zu")
    clf.fit(X)
    for metric in ["c2st_acc", "roc_auc", "ks_test", "zu"]:
        result = clf.score(X, similarity_metric=metric, test_data=X)
        if metric != "ks_test":
            assert isinstance(result, float)
        else:
            assert isinstance(result, dict)
            assert isinstance(result["statistic"], float)
            assert isinstance(result["p-value"], float)
