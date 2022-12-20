from asid.automl_small.gm import GenerativeModel
from sklearn.datasets import load_iris


def run_example():
    # Load Iris Dataset
    print("Iris dataset")
    print("")
    X = load_iris().data
    # Fit KDE model using GenerativeModel instance
    print("Fitting sklearn KDE model using GenerativeModel instance with 3 sec. Hyperopt")
    clf = GenerativeModel(gen_model_type="sklearn_kde", num_syn_samples=10, hyperopt_time=3)
    clf.fit(X)
    # Sample synthetic dataset from KDE
    print("Sampling dataset with 1000 elements from the fitted model")
    clf.sample(1000)
    # Evaluate the similarity of train and synthetic datasets with Kolmogorov-Smirnov statistic
    print("Evaluating the similarity of train and synthetic datasets with Kolmogorov-Smirnov statistic")
    clf.score(X, "ks_test")
    # Evaluate the similarity of train and synthetic datasets with Classifier Two-Sample Test
    print("Evaluating the similarity of train and synthetic datasets with Classifier Two-Sample Test")
    round(clf.score(X, "c2st_acc"), 2)
    print("")
    # Search for the optimal generative model in terms of overfitting with 'optimize' option
    print("Searching for the optimal generative model in terms of overfitting with 10 sec. Hyperopt")
    clf = GenerativeModel(gen_model_type="optimize", num_syn_samples=10, hyperopt_time=10)
    clf.fit(X)


if __name__ == "__main__":
    run_example()
