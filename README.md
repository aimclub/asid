
# ASID: Автоматическое обучение для малых и несбалансированных выборок
Библиотека ASID включает в себя инструменты автоматического обучения на малых и несбалансированных выборках в табличном формате.

Для **малых выборок** библиотека содержит алгоритм [`GenerativeModel`](https://github.com/ekplesovskaya/asid/blob/master/asid/automl_small/gm.py). Он обучает оптимальную генеративную модель, которая сэмплирует схожие синтетические выборки и не переобучается. Основные преимущества алгоритма:
* Он включает в себя 9 наиболее популярных генеративных алгоритмов для табличных данных: ядерная оценка плотности, Гауссовы смеси распределений, копулы и модели глубокого обучения;
* Инструмент достаточно прост в использовании и не требует длительного подбора гиперпараметров;
* Имеет встроенную процедуру подбора гиперпараметров с помощью Hyperopt, время работы которого может контролироваться пользователем;
* Доступны несколько метрик, которые позволяют оценить степень переобучения генеративных моделей.

Для **несбалансированных выборок** ASID содержит ансамблевый классификационный алгоритм - [`AutoBalanceBoost`](https://github.com/ekplesovskaya/asid/blob/master/asid/automl_imbalanced/abb.py) (ABB). Он сочетает в себе устойчивый ансамблевый классификатор со встроенной процедурой случайного сэмплирования (random oversampling). Основные преимущества ABB:
* Включает в себя два популярных ансамблевых алгоритма: бэггинг и бустинг;
* Содержит встроенную процедуру последовательного подбора гиперпараметров, которая позволяет получить модель высокого качества без длительного перебора гиперпараметров;
* Инструмент прост в использовании;
* Эмпирический анализ показывает, что ABB демонстрирует устойчивую работу и в среднем выдает качество выше, чем аналогичные алгоритмы.

Для **несбалансированных выборок** также разработан [`ImbalancedLearningClassifier`](https://github.com/ekplesovskaya/asid/blob/master/asid/automl_imbalanced/ilc.py), который подбирает оптимальную комбинацию балансирующей процедуры и классификатора для данной задачи. Основные преимущества инструмента:
* Включает в себя AutoBalanceBoost, а также комбинации наиболее популярных ансамблевых алгоритмов и балансирующих процедур из библиотеки imbalanced-learn;
* Инструмент достаточно прост в использовании и не требует длительного подбора гиперпараметров;
* Имеет встроенную процедуру подбора гиперпараметров для балансирующих процедур с помощью Hyperopt, время работы которого может контролироваться пользователем;
* Доступны несколько метрик оценки качества классификации.

<img src='https://user-images.githubusercontent.com/54841419/207874240-c961a176-1d29-4e7c-8107-47ff3ede8711.png' width='800'>

# Установка
Требования к версии Python: Python 3.8.

1. Установите требования из файла [requirements.txt](https://github.com/ekplesovskaya/asid/blob/master/requirements.txt)

    ```
    pip install -r requirements.txt
    ```
2. Установите библиотеку ASID
    ```
    pip install https://github.com/ekplesovskaya/asid-master.zip
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
Документация для ASID опубликована на странице [wiki](https://github.com/ekplesovskaya/asid/wiki).

Примеры использования доступны по этой [ссылке](https://github.com/ekplesovskaya/asid/tree/master/examples).
# Цитирование
Plesovskaya, Ekaterina, and Sergey Ivanov. "An Empirical Analysis of KDE-based Generative Models on Small Datasets." Procedia Computer Science 193 (2021): 442-452.
# Поддержка
Библиотека разработана при поддержке исследовательского центра [**"Сильный ИИ в промышленности"**](<https://sai.itmo.ru/>) [**Университета ИТМО**](https://itmo.ru) (г. Санкт-Петербург, Россия)
# Контакты
[Екатерина Плесовская](https://scholar.google.com/citations?user=PdydDtQAAAAJ&hl=ru), ekplesovskaya@gmail.com

[Сергей Иванов](https://scholar.google.com/citations?user=BkNV9w0AAAAJ&hl=ru), sergei.v.ivanov@gmail.com