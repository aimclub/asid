from automl_imbalanced.ilc import ImbalancedLearningClassifier
from automl_imbalanced.abb import AutoBalanceBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pathlib import Path


def run_experiments(path):
    # synthetic dataset
    print("Synthetic example")
    print("")
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=700, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # AutoBalanceBoost
    print("AutoBalanceBoost")
    clf = AutoBalanceBoost()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("F1-score for synthetic example: ", round(f1_score(y_test, pred, average="macro"), 2))
    feat_imp = clf.feature_importances()
    print("Feature importances: ", feat_imp)
    print("")
    # ImbalancedLearningClassifier
    print("ImbalancedLearningClassifier on 1 split and with 10 sec. Hyperopt")
    clf = ImbalancedLearningClassifier(split_num=1, hyperopt_time=10, eval_metric="f1_macro")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # score of the best model from ImbalancedLearningClassifier
    print("")
    print("F1-score for synthetic example: ", round(f1_score(y_test, pred, average="macro"), 2))
    # Leaderboard
    print("")
    clf.leaderboard()


if __name__ == "__main__":
    path = Path.cwd()
    run_experiments(path)
