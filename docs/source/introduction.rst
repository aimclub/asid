ASID Structure
================================

ASID library comprises autoML tools for small and imbalanced tabular datasets. ``asid/automl_small`` contains modules that allow to fit a generative model on small dataset. The main idea consists in searching for an optimal method, that generates similar synthetic datasets and does not overfit. ``asid/automl_imbalanced`` contains modules that allow to deal with imbalanced datasets in classification tasks. They include AutoBalanceBoost - an ensemble classifier specifically designed for imbalanced tasks. The key feature of this algorithm consists in a built-in sequential hyper-parameter tuning scheme. In addition to that, a tool that searches for the optimal classifier is implemented. Apart from AutoBalanceBoost, it also looks through the combinations of state-of-the-art classifiers and balancing procedures.

.. image:: img/asid_structure.png
   :width: 100%