""""""

import pickle
import os

from prophet import Prophet

CUR_PATH = os.path.dirname(__file__)
PATH_SAVE_MODEL = CUR_PATH + '\\weights'
if not os.path.exists(PATH_SAVE_MODEL):
    os.mkdir(PATH_SAVE_MODEL)


def load_model(filename: str) -> Prophet:
    """Загрузка модели"""
    filename = PATH_SAVE_MODEL + '\\' + filename
    if '.pkl' not in filename:
        filename += '.pkl'

    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model: Prophet, filename: str) -> None:
    """Сохранение модели."""
    filename = PATH_SAVE_MODEL + '\\' + filename
    if '.pkl' not in filename:
        filename += '.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(model, f)
