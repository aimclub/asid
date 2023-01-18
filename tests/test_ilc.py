from asid.automl_imbalanced.ilc import ImbalancedLearningClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
import os
import numpy as np
from scipy.stats import rankdata


def test_ilc_opt():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=500, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    clf = ImbalancedLearningClassifier(split_num=1, hyperopt_time=5)
    clf.fit(X, Y)
    assert clf.classifer_ is not None
    assert isinstance(clf.classifer_label_, str)
    assert isinstance(clf.score_, float)
    if clf.classifer_label_ != "AutoBalanceBoost":
        assert clf.scaler_ is not None
    assert len(clf.evaluated_models_scores_) != 0
    assert len(clf.evaluated_models_time_) != 0


def test_ilc_pred():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=500, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = ImbalancedLearningClassifier(split_num=1, hyperopt_time=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]
    acc = round(f1_score(y_test, pred, average="macro"), 2)
    with open("/".join(str(os.path.realpath(__file__)).split("\\")).split("tests")[
                  0] + 'tests/test_accuracy.pickle', 'rb') as f:
        acc_data = pickle.load(f)
    ilc_acc = acc_data["ILC"]
    assert acc >= ilc_acc
    if acc > ilc_acc:
        acc_data["ILC"] = acc
        with open("/".join(str(os.path.realpath(__file__)).split("\\")).split("tests")[
                      0] + 'tests/test_accuracy.pickle', "wb") as pickle_file:
            pickle.dump(acc_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_file.close()
    pred_proba = clf.predict_proba(X_test)
    assert pred_proba.shape[0] == X_test.shape[0]
    assert pred_proba.shape[1] == 4


def test_ilc_leaderboard():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=500, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    clf = ImbalancedLearningClassifier(split_num=3, hyperopt_time=0)
    clf.fit(X, Y)
    lb = clf.leaderboard()
    assert isinstance(lb, dict)
    assert len(lb["Mean score"]) != 0
    assert len(lb["Mean rank"]) != 0
    assert len(lb["Share of experiments with the first place, %"]) != 0
    assert len(lb["Average difference with the leader, %"]) != 0
    mean_dict = {}
    sub_rank_list = [[] for el in list(range(clf.split_num))]
    for el in list(clf.evaluated_models_scores_.keys()):
        mean_dict[el] = np.mean(clf.evaluated_models_scores_[el])
        for i, val in enumerate(clf.evaluated_models_scores_[el]):
            if clf.eval_metric == "log_loss":
                sub_rank_list[i].append(val)
            else:
                sub_rank_list[i].append(-val)
    if clf.eval_metric == "log_loss":
        mean_dict = {k: v for k, v in sorted(mean_dict.items(), key=lambda item: (item[1]))}
    else:
        mean_dict = {k: v for k, v in sorted(mean_dict.items(), key=lambda item: (item[1]), reverse=True)}
    leader = list(mean_dict.keys())[0]
    check_score = np.round(list(lb["Mean score"].values()), 2) == np.round(list(mean_dict.values()), 2)
    assert all(check_score)
    model_rank = []
    for i in range(len(sub_rank_list)):
        model_rank.append(rankdata(sub_rank_list[i], method='dense'))
    model_rank = np.array(model_rank)
    rank_dict = {}
    leader_share = {}
    diff_dict = {}
    for i, el in enumerate(list(clf.evaluated_models_scores_.keys())):
        rank_dict[el] = np.mean(model_rank[:, i])
        leader_share[el] = model_rank[model_rank[:, i] == 1, i].shape[0] / model_rank.shape[0] * 100
        if el != leader:
            diff = (np.array(clf.evaluated_models_scores_[el]) - np.array(
                clf.evaluated_models_scores_[leader])) / np.array(clf.evaluated_models_scores_[leader]) * 100
            diff_dict[el] = np.mean(diff)
    rank_dict = {k: v for k, v in sorted(rank_dict.items(), key=lambda item: (item[1]))}
    leader_share = {k: v for k, v in sorted(leader_share.items(), key=lambda item: (item[1]), reverse=True)}
    diff_dict = {k: v for k, v in sorted(diff_dict.items(), key=lambda item: (item[1]), reverse=True)}
    check_score = np.round(list(lb["Mean rank"].values()), 2) == np.round(list(rank_dict.values()), 2)
    assert all(check_score)
    check_score = np.round(list(lb["Share of experiments with the first place, %"].values()), 2) == np.round(
        list(leader_share.values()), 2)
    assert all(check_score)
    check_score = np.round(list(lb["Average difference with the leader, %"].values()), 2) == np.round(
        list(diff_dict.values()), 2)
    assert all(check_score)
