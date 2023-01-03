from asid.automl_small.gm import GenerativeModel
from sklearn.datasets import load_iris


def run_example():
    """
    A control test which outputs stats_kde_cv_ml as the optimal generative model with the zu metric value of 1.056
    """
    X = load_iris().data
    clf = GenerativeModel(num_syn_samples=10)
    clf.fit(X)


if __name__ == "__main__":
    run_example()