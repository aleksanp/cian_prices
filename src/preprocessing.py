import numpy as np
import pandas as pd
import random
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.set_params import set_params


python_path = os.path.join(os.getcwd())
sys.path.append(python_path)
os.environ["PYTHONPATH"] = python_path


def calculate_distance_to_sub(row, human_speed=5, bus_speed=40):
    """Функция переводит время до метро в расстояние"""

    if row.isna().any():
        return np.nan
    else:
        if row['Тип маршрута'].strip() == 'пешком':
            return row['Время'] * human_speed
        elif row['Тип маршрута'].strip() == 'на транспорте':
            return row['Время'] * bus_speed


def fit_time_to_sub(x, a, b, c):
    """Аппроксимация вычисления стоимости от расстояния до метро"""

    return a * np.exp(-b * x) + c


def fit_month_year(x, a, b):
    """Аппроксимация инфляции
    S[n] = a * ((1 + alpha)**n - 1) / (alpha)
    """
    return a * b ** (x)


def location_from_string(row):
    """Функция для преобразования строки в широту и долготу"""

    lat, lng = (row).split(',')
    lat = float(lat.replace('(', '').strip())
    lng = float(lng.replace(')', '').strip())
    return lat, lng


def get_distance(latitude, longitude, lat_center, lng_center):
    """Функция для вычисления расстояния от центра"""

    distance = np.sqrt(((latitude - lat_center) ** 2 + (longitude - lng_center) ** 2))
    return distance


def diff_days(date, start):
    """Функция преобразования даты в число дней от даты начала датасета
    start: %Y%m%d, например 20190101"""

    days = (pd.to_datetime(date) - pd.to_datetime(start, format='%Y%m%d')).days
    return days


def load_data(params):
    """Функция для загрузки исходных данных"""

    data_file = params.data
    data_path = params.data_path
    data_path = os.path.join(data_path, data_file)
    data = pd.read_csv(data_path, sep=';', skiprows=1, encoding='cp1251')
    return data


def preprocessing(data, params, distance_params=None, inflation_params=None):
    """Функция для предподготовки данных"""

    # if distance_params is None:
    #     distance_params = params.distance_params
    # if inflation_params is None:
    #     inflation_params = params.inflation_params

    data = data.drop_duplicates()
    columns = [
        'Объект продажи', 'Общая площадь', 'Этаж', 'Этажей в доме', 'Парковка',
        'Количество комнат', 'Тип дома', 'Высота потолков', 'Вид из окон',
        'Расстояние до метро', 'Адрес', 'Дата обновления'
    ]
    X = data[columns]
    y = data['Стоимость']

    # --- Объект продажи ---
    X = X[X['Объект продажи'] == 'Новостройка']

    # --- Общая площадь ---
    X['Общая площадь'] = X['Общая площадь'].str.strip('=\"').astype('float')
    # max_square = 135
    max_square = params.max_square
    X = X[X['Общая площадь'] <= max_square]

    # --- Этажей в доме ---
    mask = X['Этажей в доме'] <= 54
    X = X[mask]

    # --- Парковка ---
    X['Парковка'] = X['Парковка'].fillna('Неизвестно')
    mask = X['Парковка'] == 'подземная'
    X.loc[mask, 'Парковка'] = 'Подземная'

    # --- Количество комнат ---
    median = params.median_rooms
    X['Количество комнат'] = X['Количество комнат'].fillna(median)

    # --- Тип дома ---
    mode = params.mode_type_house
    X['Тип дома'] = X['Тип дома'].fillna(mode)

    # --- Высота потолков ---
    X['Высота потолков'] = X['Высота потолков'].fillna(0)
    X_dummy_hight = pd.get_dummies(
        pd.cut(X['Высота потолков'],
               bins=[0, 2.5, 3, 6],
               include_lowest=True,
               labels=['Высота неизвестнно', 'Высота менее 2.5-3 м', 'Высота более 3 м']
               )
    )
    X = X.join(X_dummy_hight)

    # --- Вид из окон ---
    X['Вид из окон'] = X['Вид из окон'].map(lambda row: 1 if row == 'На улицу и двор' else 0)

    # --- Геолокация ---
    locations_path = os.path.join(params.data_path, params.locations)
    X_locations = pd.read_csv(locations_path, index_col=0)
    X_locations['Адрес'] = X_locations['Адрес'].map(location_from_string)
    # lat_center = 55.7522
    # lng_center = 37.6156
    lat_center = params.lat_center
    lng_center = params.lng_center
    X['Широта'] = X_locations['Адрес'].map(lambda row: row[0])
    X['Долгота'] = X_locations['Адрес'].map(lambda row: row[1])
    X['Расстояние от центра'] = X_locations['Адрес'].map(
        lambda row: get_distance(row[0], row[1], lat_center, lng_center))

    # --- Расстояние до метро ---
    # TODO вынести отдельно, запомнить для train и использовать для test
    X_sub_distance = X['Расстояние до метро'].str.split('.', expand=True)
    X_sub_distance_time = X_sub_distance[0].str.rstrip(' мин').astype(float)
    X_sub_distance_time.name = 'Время до метро'
    mask = X_sub_distance_time.isna()
    mask.name = 'Пропущено время'
    X_sub_distance.columns = 'Время', 'Тип маршрута'
    X_sub_distance['Время'] = (X_sub_distance['Время'][~mask]
                               .map(lambda x: int(x.split(' ')[0])))
    X_sub_distance['Расстояние до метро'] = X_sub_distance.apply(calculate_distance_to_sub, axis=1)
    # left_border = 1400
    # right_border = 4500
    left_border = params.sub_distance_left_border
    right_border = params.sub_distance_right_border
    X['Расстояние до метро'] = X_sub_distance['Расстояние до метро'].map(
        lambda row: random.randint(left_border, right_border) if pd.isna(row) else row)
    # X['Стоимость от расстояния'] = X['Расстояние до метро'].map(
    #     lambda x, pars=distance_params: fit_time_to_sub(x, *pars))

    # --- Инфляция ---
    # TODO вынести отдельно, запомнить для train и использовать для test
    start_date = params.start_date
    X['Дней'] = X['Дата обновления'].map(lambda x: diff_days(x, start_date))
    # X['Стоимость с инфляцией'] = X['Дней'].map(lambda row: fit_month_year(row, *inflation_params))

    y = y[X.index]

    return X, y


def feature_encoding(features, encoder, x_train, x_test, x_valid=None,
                     y_train=None, onehot_cols=False, *args, **kwargs):
    """Универсальная функция для кодирования признаков"""

    encoder = encoder(*args, **kwargs)

    if y_train is not None:
        encoder.fit(x_train[features], y_train)
    else:
        encoder.fit(x_train[features])
    # --- train ---
    cols_train = pd.DataFrame(encoder.transform(x_train[features]))
    cols_train.index = x_train[features].index
    if onehot_cols:
        cols_train.columns = [item for sublist in encoder.categories_ for item in sublist]
    else:
        cols_train.columns = x_train[features].columns
    # --- test ---
    cols_test = pd.DataFrame(encoder.transform(x_test[features]))
    cols_test.index = x_test[features].index
    if onehot_cols:
        cols_test.columns = [item for sublist in encoder.categories_ for item in sublist]
    else:
        cols_test.columns = x_test[features].columns
    # --- valid ---
    if not x_valid is None:
        cols_valid = pd.DataFrame(encoder.transform(x_valid[features]))
        cols_valid.index = x_valid[features].index
        if onehot_cols:
            cols_valid.columns = [item for sublist in encoder.categories_ for item in sublist]
        else:
            cols_valid.columns = x_valid[features].columns
        return cols_train, cols_test, cols_valid

    return cols_train, cols_test


def all_features_encoding(x_train, x_test, x_valid, cat_features, num_features,
                          num_encoder=StandardScaler, cat_encoder=OneHotEncoder):
    """Функция для кодирования признаков"""

    x_train_num, x_test_num, x_valid_num = feature_encoding(num_features, num_encoder, x_train, x_test, x_valid)
    x_train_cat, x_test_cat, x_valid_cat = feature_encoding(cat_features, cat_encoder, x_train, x_test, x_valid,
                                                            onehot_cols=True, sparse=False)
    x_train = x_train_num.join(x_train_cat)
    x_test = x_test_num.join(x_test_cat)
    x_valid = x_valid_num.join(x_valid_cat)

    return x_train, x_test, x_valid


if __name__ == '__main__':
    params_path = 'params.yaml'
    params = set_params(params_path)
    data_all = load_data(params)
    X, y = preprocessing(data_all, params)
    print(f"{X} \n {'=' * 20} \n {y}")
