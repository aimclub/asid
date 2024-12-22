from asid.automl_small.gm import GenerativeModel
from sklearn.datasets import load_iris, make_classification
from asid.automl_imbalanced.abb import AutoBalanceBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from asid.automl_imbalanced.ilc import ImbalancedLearningClassifier


def test_first_example():
    X = load_iris().data
    genmod = GenerativeModel()
    genmod.fit(X)
    genmod.sample(1000)
    assert True


def test_second_example():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=700, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = AutoBalanceBoost()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = f1_score(y_test, pred, average="macro")
    assert True


def test_third_example():
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=700, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = ImbalancedLearningClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = f1_score(y_test, pred, average="macro")
    assert True
