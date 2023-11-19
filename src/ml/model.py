""""""

import pickle
import os
from typing import Union, Optional
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from prophet import Prophet
import pandas as pd


CUR_PATH = os.path.dirname(__file__)
PATH_SAVE_MODEL = CUR_PATH + '/weights'
if not os.path.exists(PATH_SAVE_MODEL):
    os.mkdir(PATH_SAVE_MODEL)


class ModelProphet:
    """Класс модели."""

    def __init__(
            self,
            model: Union[str, Prophet],
            cap: Optional[float] = None,
            floor: Optional[float] = None,
            start_date: date = date(2021, 11, 30),
    ):
        if isinstance(model, Prophet):
            self.model = model
        elif isinstance(model, str):
            self.model = self.load(model)

        self.cap = cap
        self.floor = floor

        self.start_date = start_date

    def save(self) -> None:
        """Сохранение модели."""
        filename = PATH_SAVE_MODEL + '/' + filename
        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filename: str) -> Prophet:
        """Загрузка модели"""
        filename = PATH_SAVE_MODEL + '/' + filename
        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'rb') as f:
            model = pickle.load(f)
        self.model = model
        return model

    def predict(self, year: int, month: int, day: int) -> pd.DataFrame:
        """"""
        end_date = date(year, month, day)

        # Сразу добавляем стартовую дату
        periods = [self.start_date]

        new_date = self.start_date
        while new_date < end_date:
            new_date += relativedelta(months=1) 
            periods.append(new_date)

        if end_date not in periods:
            periods.append(end_date)

        data = pd.DataFrame({'ds': periods})

        if self.floor is not None:
            data['floor'] = self.floor
        if self.cap is not None:
            data['cap'] = self.cap

        pred = self.model.predict(data)
        pred.index = data.index
        return pred[['ds', 'yhat']]


if __name__ == '__main__':
    model = ModelProphet('model_growth_clients_rko.pkl', floor=0, cap=100)
    print(model)
    print(model.predict(2024, 2, 27))
