from asid.automl_imbalanced.abb import AutoBalanceBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import f1_score
import os


def test_abb_fit():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=500, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    clf = AutoBalanceBoost()
    clf.fit(X, Y)
    assert clf.ensemble_ is not None
    assert len(clf.param_) != 0


def test_abb_pred():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=500, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = AutoBalanceBoost()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]
    acc = round(f1_score(y_test, pred, average="macro"), 2)
    with open("/".join(str(os.path.realpath(__file__)).split("\\")).split("tests")[
                  0] + 'tests/test_accuracy.pickle', 'rb') as f:
        acc_data = pickle.load(f)
    abb_acc = acc_data["ABB"]
    assert acc >= abb_acc
    if acc > abb_acc:
        acc_data["ABB"] = acc
        with open("/".join(str(os.path.realpath(__file__)).split("\\")).split("tests")[
                      0] + 'tests/test_accuracy.pickle', "wb") as pickle_file:
            pickle.dump(acc_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_file.close()
    pred_proba = clf.predict_proba(X_test)
    assert pred_proba.shape[0] == X_test.shape[0]
    assert pred_proba.shape[1] == 4


def test_abb_feat():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=500, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = AutoBalanceBoost()
    clf.fit(X_train, y_train)
    feat_imp = clf.feature_importances()
    assert feat_imp.shape[0] == X_test.shape[1]
    assert round(sum(feat_imp)) == 1
