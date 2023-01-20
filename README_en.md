# ASID: AutoML for Small and Imbalanced Datasets
ASID library comprises autoML tools for small and imbalanced tabular datasets.

For **small datasets** we propose a [`GenerativeModel`](https://github.com/ekplesovskaya/asid/blob/master/asid/automl_small/gm.py) estimator that searches for an optimal generative algorithm, which outputs similar synthetic samples and does not overfit. Main features of this tool:
* It includes 9 popular generative approaches for small tabular datasets such as kernel density estimation, gaussian mixture models, copulas and deep learning models;
* It is easy-to-use and does not require time-consuming tuning;
* It includes a Hyperopt tuning procedure, which could be controlled by a runtime parameter;
* Several overfitting indicators are available.

For **imbalanced datasets** ASID library includes a tailored ensemble classifier - [`AutoBalanceBoost`](https://github.com/ekplesovskaya/asid/blob/master/asid/automl_imbalanced/abb.py). It combines a consistent ensemble classifier with the embedded random oversampling technique. ABB key features include:
* It exploits both popular ensemble approaches: bagging and boosting;
* It comprises an embedded sequential parameter tuning scheme, which allows to get the high accuracy without time-consuming tuning;
* It is easy-to-use and does not require time-consuming tuning;
* Empirical analysis shows that ABB demonstrates a robust performance and on average outperforms its competitors.

For **imbalanced datasets** we also propose an [`ImbalancedLearningClassifier`](https://github.com/ekplesovskaya/asid/blob/master/asid/automl_imbalanced/ilc.py) estimator that searches for an optimal classifier for a given imbalanced task. Main features of this tool:
* It includes AutoBalanceBoost and combinations of SOTA ensemble algorithms and balancing procedures from imbalanced-learn library;
* It is easy-to-use and does not require time-consuming tuning;
* It includes a Hyperopt tuning procedure for balancing procedures, which could be controlled by a runtime parameter;
* Several classification accuracy metrics are available.

<img src='https://user-images.githubusercontent.com/54841419/213721694-89b4b9a9-97e7-43dc-8beb-ecaecb506fe6.png' width='1000'>

# How to install
Requirements: Python 3.8.

1. Install requirements from [requirements.txt](https://github.com/ekplesovskaya/asid/blob/master/requirements.txt)

    ```
    pip install -r requirements.txt
    ```
2. Install ASID library as a package
    ```
    pip install https://github.com/ekplesovskaya/asid-master.zip
    ```
# Usage examples
Fitting a GenerativeModel instance on small sample and generating a synthetic dataset:
```python
from asid.automl_small.gm import GenerativeModel
from sklearn.datasets import load_iris

X = load_iris().data
genmod = GenerativeModel()
genmod.fit(X)
genmod.sample(1000)
```
Fitting an AutoBalanceBoost classifier on imbalanced dataset:
```python
from asid.automl_imbalanced.abb import AutoBalanceBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                           n_clusters_per_class=2, flip_y=0.05, n_samples=700, random_state=45,
                           weights=(0.7, 0.2, 0.05, 0.05))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = AutoBalanceBoost()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = f1_score(y_test, pred, average="macro")
```
Choosing an optimal classification pipeline with ImbalancedLearningClassifier for imbalanced dataset (searches through AutoBalanceBoost and combinations of SOTA ensemble algorithms and balancing procedures from imbalanced-learn library):
```python
from asid.automl_imbalanced.ilc import ImbalancedLearningClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X, Y = make_classification(n_classes=4, n_features=6, n_redundant=2, n_repeated=0, n_informative=4,
                           n_clusters_per_class=2, flip_y=0.05, n_samples=700, random_state=45,
                           weights=(0.7, 0.2, 0.05, 0.05))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = ImbalancedLearningClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = f1_score(y_test, pred, average="macro")
```
# Benchmarks
Results or empirical experiments with ASID algorithms are available [here](https://github.com/ekplesovskaya/asid/wiki/5.-Benchmarks).
# Documentation
Documentation about ASID could be found in [wiki](https://github.com/ekplesovskaya/asid/wiki).

Examples of usage could be obtained from [examples](https://github.com/ekplesovskaya/asid/tree/master/examples).
# Citation
Plesovskaya, Ekaterina, and Sergey Ivanov. "An Empirical Analysis of KDE-based Generative Models on Small Datasets." Procedia Computer Science 193 (2021): 442-452.
# Supported by
The study is supported by Research Center [**Strong Artificial Intelligence in Industry**](<https://sai.itmo.ru/>)
of [**ITMO University**](https://itmo.ru) (Saint Petersburg, Russia)
# Contacts
[Ekaterina Plesovskaya](https://scholar.google.com/citations?user=PdydDtQAAAAJ&hl=ru), ekplesovskaya@gmail.com

[Sergey Ivanov](https://scholar.google.com/citations?user=BkNV9w0AAAAJ&hl=ru), sergei.v.ivanov@gmail.com