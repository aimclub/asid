from asid.automl_imbalanced.abb import AutoBalanceBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def run_example():
    """
    A control test which outputs F1-score value of 0.6
    """
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


if __name__ == "__main__":
    run_example()