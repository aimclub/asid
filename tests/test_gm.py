from asid.automl_small.gm import GenerativeModel
from sklearn.datasets import load_iris
import pickle
import os


def test_gm_opt():
    X = load_iris().data
    clf = GenerativeModel(gen_model_type="optimize", num_syn_samples=10, hyperopt_time=5, similarity_metric="zu")
    clf.fit(X)
    with open("/".join(str(os.path.realpath(__file__)).split("\\")).split("tests")[
                  0] + 'tests/test_accuracy.pickle', 'rb') as f:
        acc_data = pickle.load(f)
    gm_acc = acc_data["GM"]
    assert clf.score_ <= gm_acc
    if clf.score_ < gm_acc:
        acc_data["GM"] = round(clf.score_, 2)
        with open("/".join(str(os.path.realpath(__file__)).split("\\")).split("tests")[
                      0] + 'tests/test_accuracy.pickle', "wb") as pickle_file:
            pickle.dump(acc_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_file.close()
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
            if metric == "c2st_acc":
                assert round(result, 2) == 0.5
            elif metric == "roc_auc":
                assert round(result, 2) >= 0.93
            elif metric == "zu":
                assert round(result, 2) == 14.98
        else:
            assert isinstance(result, dict)
            assert isinstance(result["statistic"], float)
            assert isinstance(result["p-value"], float)
            assert round(result["statistic"], 2) == 0.12
            assert round(result["p-value"], 2) == 0.0
