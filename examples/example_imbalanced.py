from asid.automl_imbalanced.ilc import ImbalancedLearningClassifier
from asid.automl_imbalanced.abb import AutoBalanceBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def run_example():
    # Create a synthetic dataset
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=700, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Fit AutoBalanceBoost and evaluate its performance on synthetic dataset
    print("AutoBalanceBoost")
    clf = AutoBalanceBoost()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("AutoBalanceBoost F1-score for synthetic example: ", round(f1_score(y_test, pred, average="macro"), 2))
    # Calculate feature importances of AutoBalanceBoost
    feat_imp = clf.feature_importances()
    print("Feature importances: ", feat_imp)
    print("")
    # Fit ImbalancedLearningClassifier on 1 split and with 10 sec. Hyperopt
    print("ImbalancedLearningClassifier on 1 split and with 10 sec. Hyperopt")
    clf = ImbalancedLearningClassifier(split_num=1, hyperopt_time=10, eval_metric="f1_macro")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # Evaluate ImbalancedLearningClassifier performance on synthetic dataset
    print("")
    print("ImbalancedLearningClassifier F1-score for synthetic example: ",
          round(f1_score(y_test, pred, average="macro"), 2))
    # Calculate the leaderboard statistic
    print("")
    clf.leaderboard()


if __name__ == "__main__":
    run_example()
