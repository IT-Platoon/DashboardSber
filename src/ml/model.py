""""""

import pickle
import os
from typing import Union

from prophet import Prophet
import pandas as pd


CUR_PATH = os.path.dirname(__file__)
PATH_SAVE_MODEL = CUR_PATH + '\\weights'
if not os.path.exists(PATH_SAVE_MODEL):
    os.mkdir(PATH_SAVE_MODEL)


def ModelProphet(ABC):
    """Класс модели."""

    def __init__(self, model: Union[str, Prophet]):
        if isinstance(model, Prophet):
            self.model = model
        elif isinstance(model, str):
            self.model = self.load(model)

    def save(self) -> None:
        """Сохранение модели."""
        filename = PATH_SAVE_MODEL + '\\' + filename
        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filename: str) -> Prophet:
        """Загрузка модели"""
        filename = PATH_SAVE_MODEL + '\\' + filename
        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'rb') as f:
            model = pickle.load(f)
        self.model = model
        return model

    def predict(data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """"""

