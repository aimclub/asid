from asid.automl_imbalanced.ilc import ImbalancedLearningClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


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
    pred_proba = clf.predict_proba(X_test)
    assert pred_proba.shape[0] == X_test.shape[0]
    assert pred_proba.shape[1] == 4


def test_ilc_leaderboard():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=500, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    clf = ImbalancedLearningClassifier(split_num=1, hyperopt_time=0)
    clf.fit(X, Y)
    lb = clf.leaderboard()
    assert isinstance(lb, dict)
    assert len(lb["Mean score"]) != 0
    assert len(lb["Mean rank"]) != 0
    assert len(lb["Share of experiments with the first place, %"]) != 0
    assert len(lb["Average difference with the leader, %"]) != 0
