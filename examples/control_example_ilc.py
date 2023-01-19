from asid.automl_imbalanced.ilc import ImbalancedLearningClassifier
from sklearn.datasets import make_classification


def run_example():
    """
    A control test which outputs F1-score value of 0.73
    """
    # Create a synthetic dataset
    X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                               n_clusters_per_class=2, flip_y=0.05, n_samples=700, random_state=45,
                               weights=(0.7, 0.2, 0.05, 0.05))
    clf = ImbalancedLearningClassifier(split_num=1)
    clf.fit(X, Y)


if __name__ == "__main__":
    run_example()