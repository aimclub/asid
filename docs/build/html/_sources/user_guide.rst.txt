User Guide
================================

Synthetic dataset generation in ASID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To fit a generative model with ASID import a ``GenerativeModel`` instance.

.. code-block:: python

   from asid.automl_small.gm import GenerativeModel

There are several modes of generative model estimation available in GenerativeModel class. Firstly, a certain type of generative algorithm could be chosen. For example, a scikit-learn implementation of KDE could be fitted.

.. code-block:: python

    from sklearn.datasets import load_iris

    X = load_iris().data
    genmod = GenerativeModel(gen_model_type="sklearn_kde")
    genmod.fit(X)

The option ``optimize`` allows to search through a number of generative algorithms and returns an optimal option in terms of overfitting. It is also possible to control the number of synthetic samples that are used to evaluate overfitting with ``num_syn_samples`` parameter, and Hyperopt time for generative model hyper-parameters optimization with ``hyperopt_time`` parameter.

.. code-block:: python

    genmod = GenerativeModel(gen_model_type="optimize", num_syn_samples=10, hyperopt_time=10)
    genmod.fit(X)

After the model is fitted it is possible to generate synthetic datasets of the required size or evaluate the similarity of model's samples with the train dataset with one of the implemented indicators.

.. code-block:: python

    genmod.sample(1000)
    genmod.score(X, "ks_test")

Imbalanced learning in ASID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AutoBalanceBoost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import an ``AutoBalanceBoost`` instance from an ``automl_imbalanced`` module.

.. code-block:: python

    from asid.automl_imbalanced.abb import AutoBalanceBoost

AutoBalanceBoost is an easy-to-use tool, that allows to obtain a high-quality model without time-consuming hyper-parameters tuning.

.. code-block:: python

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

In addition to that, feature importances of AutoBalanceBoost could be also calculated.

.. code-block:: python

    feat_imp = clf.feature_importances()

Choosing an imbalanced learning classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import an ``ImbalancedLearningClassifier`` instance from an ``automl_imbalanced`` module.

.. code-block:: python

    from asid.automl_imbalanced.ilc import ImbalancedLearningClassifier

``ImbalancedLearningClassifier`` looks through the combinations of state-of-the-art classifiers and balancing procedures, compares their result with AutoBalanceBoost and chooses the best classifier. Users could control the number of splits (``split_num``) that are used to evaluate the classifiers performance, Hyperopt time (``hyperopt_time``) for balancing algorithms hyper-parameters optimization and classification score metric (``eval_metric``).

.. code-block:: python

    clf = ImbalancedLearningClassifier(split_num=50, hyperopt_time=10, eval_metric="f1_macro")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = f1_score(y_test, pred, average="macro")

The leaderboard statistics is also available once ``ImbalancedLearningClassifier`` is fitted. It includes sorted lists in accordance with the following indicators: "Mean score", "Mean rank", "Share of experiments with the first place, %", "Average difference with the leader, %".

.. code-block:: python

    clf.leaderboard()










