from asid.automl_small.gm import GenerativeModel
from sklearn.datasets import load_iris
from pathlib import Path


def run_experiments(path):
    # Iris dataset
    print("Iris dataset")
    print("")
    X = load_iris().data
    # Fitting KDE model using GenerativeModel instance
    print("Fitting sklearn KDE model using GenerativeModel instance with 3 sec. Hyperopt")
    clf = GenerativeModel(gen_model_type="sklearn_kde", num_syn_samples=10, hyperopt_time=3)
    clf.fit(X)
    # Sampling from KDE
    print("Sampling dataset with 1000 elements from the fitted model")
    clf.sample(1000)
    # Evaluating the similarity of train and synthetic datasets with Kolmogorov-Smirnov statistic
    print("Evaluating the similarity of train and synthetic datasets with Kolmogorov-Smirnov statistic")
    clf.score(X, "ks_test")
    print("Evaluating the similarity of train and synthetic datasets with Classifier Two-Sample Test")
    round(clf.score(X, "c2st_acc"), 2)
    print("")
    # Searching for the optimal generative model in terms of overfitting
    print("Searching for the optimal generative model in terms of overfitting with 10 sec. Hyperopt")
    clf = GenerativeModel(gen_model_type="optimize", num_syn_samples=10, hyperopt_time=10)
    clf.fit(X)


if __name__ == "__main__":
    path = Path.cwd()
    run_experiments(path)
