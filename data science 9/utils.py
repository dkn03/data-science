from typing import Iterable, Union, Literal, Mapping, Sequence
from sklearn.model_selection import GridSearchCV

def grid_search(model, params: Union[Mapping, Sequence[dict]], x_train: Iterable, y_train: Iterable, scoring=None):
    """
    Отбор лучших параметров модели при помощи GridSearch
    :param model: Класс модели
    :param params: Настраиваемые параметры модели
    :param x_train: Данные для классификации
    :param y_train: Выходные данные классификатора
    """

    grid = GridSearchCV(model, param_grid=params, scoring=scoring)
    grid.fit(x_train, y_train)
    return grid.best_estimator_