import pandas as pd

from typing import Iterable, Union, Literal, Mapping, Sequence
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.stats import sigmaclip
from numpy import isnan
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error

def encode_onehot_categorical(dataframe: pd.DataFrame, columns: Iterable):
    """
    Кодирует столбцы columns с помощью OneHotEncoder, удаляя первоначальные столбцы.
    :param frame: таблица DataFrame
    :param columns: кодируемые столбцы
    :return: новая таблица
    """
    encoder = OneHotEncoder(drop='first')
    for column in columns:
        transformed = encoder.fit_transform(dataframe[[column]]).toarray()
        new_frame = pd.DataFrame(transformed, columns=encoder.get_feature_names_out(), index=dataframe.index)
        dataframe = dataframe.join(new_frame)
    dataframe = dataframe.drop(columns=columns)
    return dataframe

def sigma_clip(frame, column):
    """
    Удаление выбросов из столбца методом сигм
    :param frame: Таблица DataFrame библиотеки Pandas
    :param column: Название столбца таблицы
    :return: Новый столбец с удаленными выбросами
    """
    clipped = sigmaclip(frame[column].dropna())
    lower_border, upper_border = clipped[1], clipped[2]
    return frame[(frame[column] >= lower_border)
                 & (frame[column] <= upper_border)
                 | (isnan(frame[column]))].reset_index(drop=True)
def preprocess_df(dataframe: pd.DataFrame, output, categorical: Iterable, numeric: Iterable = None, scaler: Literal['minmax', 'standard', None] = 'minmax', 
                  resample: bool = False, to_drop=None, ):
    """
    Предобработка таблицы dataframe
    :param dataframe: обрабатываемая таблица dataframe
    :param output: столбец выходных данных
    :param categorical: категориальные столбцы таблицы
    :param numeric: числовые столбцы таблицы
    :param scaler: метод масштабирования. 
    'minmax' - нормализация от 0 до 1, 'standard' - стандартизация от -1 до 1, None - без масштабирования
    :param resample: Указывает, будет ли применятся Oversampling для данных. По умолчанию False, работает только для задачи классификации
    :param to_drop: удаляемые столбцы
    :return: новая таблица
    """
    dataframe.drop(columns=to_drop, inplace=True)
    if not numeric:
        numeric = dataframe.select_dtypes(include='number').columns
    for column in numeric:
        dataframe = sigma_clip(dataframe, column)
    dataframe = encode_onehot_categorical(dataframe, categorical)
    match scaler:
        case 'minmax':
            dataframe[numeric] = MinMaxScaler().fit_transform(dataframe[numeric])
        case 'standard':
            dataframe[numeric] = StandardScaler().fit_transform(dataframe[numeric])
        case _:
            pass
    if resample:
        x_resampled, y_resampled = SMOTE().fit_resample(dataframe.drop(columns=output), dataframe[output])
        dataframe = x_resampled.join(y_resampled)
    return dataframe

def grid_search(model, params: Union[Mapping, Sequence[dict]], x_train: Iterable, y_train: Iterable):
    """
    Отбор лучших параметров модели при помощи GridSearch
    :param model: Класс модели
    :param params: Настраиваемые параметры модели
    :param x_train: Данные для классификации
    :param y_train: Выходные данные классификатора
    """
    grid = GridSearchCV(model, param_grid=params)
    grid.fit(x_train, y_train)
    return grid.best_estimator_

def get_classification_metrics(y_true: Iterable, y_predicted: Iterable):
    """"
    Вычисление основных метрик для предсказания классов
    :param y_true: Реальные данные 
    :param y_predicted: Данные, предсказанные моделью
    """
    return {"Accuracy": accuracy_score(y_true, y_predicted),
            "Precision": precision_score(y_true, y_predicted),
            "Recall": recall_score(y_true, y_predicted),
            "F1": f1_score(y_true, y_predicted),
            "Roc auc score": roc_auc_score(y_true, y_predicted)}

def get_regression_metrics(y_true: Iterable, y_predicted: Iterable):
    """
    Вычисление основных метрик регрессии
    :param y_true: Реальные данные 
    :param y_predicted: Данные, предсказанные моделью 
    """
    return {"RMSE": mean_squared_error(y_true, y_predicted, squared=False),
            "R2": r2_score(y_true, y_predicted)}


