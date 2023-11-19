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
            old_date: Union[pd.DataFrame, str],
            cap: Optional[float] = None,
            floor: Optional[float] = None,
    ):
        if isinstance(model, Prophet):
            self.model = model
        elif isinstance(model, str):
            self.model = self.load(model)

        if isinstance(old_date, pd.DataFrame):
            self.old_date = old_date
        elif isinstance(old_date, str):
            old_date = CUR_PATH + '/data/' + old_date
            self.old_date = pd.read_csv(old_date)

        self.cap = cap
        self.floor = floor

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
        date_last_list = self.old_date.ds.iloc[-1].split('-')
        date_last = date(int(date_last_list[0]), int(date_last_list[1]), int(date_last_list[2]))
        periods = [date_last + relativedelta(months=1)]

        new_date = periods[0]
        while new_date < end_date:
            new_date += relativedelta(months=1) 
            periods.append(new_date)
        print(date_last)
        print(periods)

        if end_date not in periods:
            periods.append(end_date)

        data_for_pred = pd.DataFrame({'ds': periods})
        if self.floor is not None:
            data_for_pred['floor'] = self.floor
        if self.cap is not None:
            data_for_pred['cap'] = self.cap

        pred = self.model.predict(data_for_pred)
        pred = pred[['ds', 'yhat']]
        pred.columns = ['ds', 'y']

        df_final = pd.concat([self.old_date, pred], ignore_index=True)
        df_final.ds = df_final.ds.map(lambda x: x if isinstance(x, str ) else date.isoformat(x))
        return df_final


if __name__ == '__main__':
    model = ModelProphet(
        'model_growth_clients_eq.pkl',
        'data_growth_clients_eq.csv',

        floor=159,
        cap=8546
    )
    print(model)
    print(model.predict(2024, 2, 27))
