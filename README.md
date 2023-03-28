
# ASID: Автоматическое обучение для малых и несбалансированных выборок
[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg)](https://en.itmo.ru/en/)

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Documentation](https://github.com/aimclub/asid/actions/workflows/docs.yml/badge.svg)](https://aimclub.github.io/asid/docs/build/html/index.html)
[![Tests](https://github.com/aimclub/asid/actions/workflows/test.yml/badge.svg)](https://github.com/aimclub/asid/actions/workflows/test.yml)
[![Mirror](https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162)](https://gitlab.actcognitive.org/itmo-sai-code/asid/)

Библиотека ASID включает в себя инструменты автоматического обучения на малых и несбалансированных выборках в табличном формате.

Для **малых выборок** библиотека содержит алгоритм [`GenerativeModel`](https://gitlab.actcognitive.org/itmo-sai-code/asid/-/blob/master/asid/automl_small/gm.py). Он обучает оптимальную генеративную модель, которая сэмплирует схожие синтетические выборки и не переобучается. Основные преимущества алгоритма:
* Он включает в себя 9 наиболее популярных генеративных алгоритмов для табличных данных: ядерная оценка плотности, Гауссовы смеси распределений, копулы и модели глубокого обучения;
* Инструмент достаточно прост в использовании и не требует длительного подбора гиперпараметров;
* Имеет встроенную процедуру подбора гиперпараметров с помощью Hyperopt, время работы которого может контролироваться пользователем;
* Доступны несколько метрик, которые позволяют оценить степень переобучения генеративных моделей.

Для **несбалансированных выборок** ASID содержит ансамблевый классификационный алгоритм - [`AutoBalanceBoost`](https://gitlab.actcognitive.org/itmo-sai-code/asid/-/blob/master/asid/automl_imbalanced/abb.py) (ABB). Он сочетает в себе устойчивый ансамблевый классификатор со встроенной процедурой случайного сэмплирования (random oversampling). Основные преимущества ABB:
* Включает в себя два популярных ансамблевых алгоритма: бэггинг и бустинг;
* Содержит встроенную процедуру последовательного подбора гиперпараметров, которая позволяет получить модель высокого качества без длительного перебора гиперпараметров;
* Инструмент прост в использовании;
* Эмпирический анализ показывает, что ABB демонстрирует устойчивую работу и в среднем выдает качество выше, чем аналогичные алгоритмы.

Для **несбалансированных выборок** также разработан [`ImbalancedLearningClassifier`](https://gitlab.actcognitive.org/itmo-sai-code/asid/-/blob/master/asid/automl_imbalanced/ilc.py), который подбирает оптимальную комбинацию балансирующей процедуры и классификатора для данной задачи. Основные преимущества инструмента:
* Включает в себя AutoBalanceBoost, а также комбинации наиболее популярных ансамблевых алгоритмов и балансирующих процедур из библиотеки imbalanced-learn;
* Инструмент достаточно прост в использовании и не требует длительного подбора гиперпараметров;
* Имеет встроенную процедуру подбора гиперпараметров для балансирующих процедур с помощью Hyperopt, время работы которого может контролироваться пользователем;
* Доступны несколько метрик оценки качества классификации.

<img src='https://user-images.githubusercontent.com/54841419/213721694-89b4b9a9-97e7-43dc-8beb-ecaecb506fe6.png' width='1000'>

# Установка
Требования к версии Python: Python 3.8.

1. Установите требования из файла [requirements.txt](https://gitlab.actcognitive.org/itmo-sai-code/asid/-/blob/master/requirements.txt)

    ```
    pip install -r requirements.txt
    ```
2. Установите библиотеку ASID
    ```
    pip install  https://github.com/aimclub/asid/archive/refs/heads/master.zip
    ```
# Примеры использования
Обучение модели по малой выборке с помощью GenerativeModel и генерация синтетического датасета:
```python
from asid.automl_small.gm import GenerativeModel
from sklearn.datasets import load_iris

X = load_iris().data
genmod = GenerativeModel()
genmod.fit(X)
genmod.sample(1000)
```
Обучение модели AutoBalanceBoost на несбалансированном датасете:
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
Подбор оптимальной схемы классификации для несбалансированного датасета с помощью ImbalancedLearningClassifier (поиск производится среди AutoBalanceBoost, а также комбинаций наиболее популярных ансамблевых алгоритмов и балансирующих процедур из библиотеки imbalanced-learn):
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
# Бенчмарки
Результаты эмпирического анализа алгоритмов ASID на различных датасетах доступны [здесь](https://github.com/ekplesovskaya/asid/wiki/5.-Benchmarks).

# Документация
Документация для ASID доступна по этому [адресу](https://aimclub.github.io/asid/docs/build/html/index.html).

Примеры использования доступны по этой [ссылке](https://gitlab.actcognitive.org/itmo-sai-code/asid/-/tree/master/examples).

# Цитирование
ГОСТ:

> Plesovskaya, Ekaterina, and Sergey Ivanov. "An Empirical Analysis of KDE-based Generative Models on Small Datasets." Procedia Computer Science 193 (2021): 442-452.

Bibtex:

```bibtex
@article{plesovskaya2021empirical,
  title={An empirical analysis of KDE-based generative models on small datasets},
  author={Plesovskaya, Ekaterina and Ivanov, Sergey},
  journal={Procedia Computer Science},
  volume={193},
  pages={442--452},
  year={2021},
  publisher={Elsevier}
}
```

# Поддержка
Исследование проводится при поддержке [Исследовательского центра сильного искусственного интеллекта в промышленности](https://sai.itmo.ru/) [Университета ИТМО](https://itmo.ru/) в рамках мероприятия программы центра: Разработка и испытания экспериментального образца библиотеки алгоритмов сильного ИИ в части базовых алгоритмов оценки качества и автоматической адаптации моделей машинного обучения под сложность задачи и размер выборки на основе генеративного синтеза комплексных цифровых объектов 

<a href='https://sai.itmo.ru/'>
  <img src='https://gitlab.actcognitive.org/itmo-sai-code/organ/-/raw/main/docs/AIM-Strong_Sign_Norm-01_Colors.svg' width='200'>
</a>

# Контакты
[Екатерина Плесовская](https://scholar.google.com/citations?user=PdydDtQAAAAJ&hl=ru), ekplesovskaya@gmail.com

[Сергей Иванов](https://scholar.google.com/citations?user=BkNV9w0AAAAJ&hl=ru), sergei.v.ivanov@gmail.com