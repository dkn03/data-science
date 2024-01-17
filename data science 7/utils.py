from sklearn.preprocessing import OneHotEncoder
from scipy.stats import sigmaclip
import pandas as pd
import numpy as np

def encode_onehot_categorical(frame, columns):
    """
    Кодирует столбцы columns с помощью OneHotEncoder, удаляя первоначальные столбцы.
    :param frame: таблица DataFrame
    :param columns: кодируемые столбцы
    :return: новая таблица
    """
    encoder = OneHotEncoder()
    for column in columns:
        transformed = encoder.fit_transform(frame[[column]]).toarray()[:, 1:]
        new_frame = pd.DataFrame(transformed, columns=encoder.get_feature_names_out()[1:], index=frame.index)
        frame = frame.join(new_frame)
    frame = frame.drop(columns=columns)
    return frame

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
                 | (np.isnan(frame[column]))].reset_index(drop=True)


def remove_weakly_correlated(dataframe: pd.DataFrame, output_column: str, threshold : int=0.1):
    """
    Удаление признаков, слабо коррелирующих с целевой переменной
    :param dataframe: Таблица DataFrame библиотеки Pandas
    :param output_column: название столбца с целевой переменной
    :param threshold: минимально допустимая корреляция, 0.1 по умолчанию
    """
    corr_df = dataframe.corr()
    corr_df = corr_df.loc[abs(corr_df[output_column]) > threshold]
    columns_left = corr_df.index
    return dataframe[columns_left]


def remove_highly_correlated(dataframe: pd.DataFrame, output_column : str,  threshold: int = 0.7):
    """
    Удаление сильно коррелирующих друг с другом признаков. 
    Из двух признаков оставляет тот, который сильнее коррелирует с целевой переменной
    :param dataframe: Таблица DataFrame библиотеки Pandas
    :param output_column: название столбца с целевой переменной
    :param threshold: минимальная корреляция, с которой удаляются столбцы
    """
    corr_df = dataframe.corr()
    removed = set()
    for row in corr_df.drop(index=output_column, columns=output_column).iterrows():
        row = row[0]
        for column in corr_df.drop(index=output_column, columns=output_column).columns:
            if threshold <= corr_df[row][column] < 1:
                removed.add(min([row, column], key=lambda x: abs(corr_df[x][output_column])))
    corr_df = corr_df.drop(index=removed, columns=removed)
    return dataframe[corr_df.columns]