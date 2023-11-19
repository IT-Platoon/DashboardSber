from datetime import datetime
from typing import Union, Literal
#Plotting libraries
import altair as alt
# plt.style.use('seaborn-white')

#statistics libraries
import statsmodels.api as sm
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import anderson
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import month_plot, seasonal_plot, plot_acf, plot_pacf, quarter_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox as ljung
from statsmodels.tsa.statespace.tools import diff as diff
import pmdarima as pm
from pmdarima import ARIMA, auto_arima
from scipy import signal
from scipy.stats import shapiro
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
import ruptures as rpt


import warnings
warnings.filterwarnings("ignore")
np.random.seed(786)


def fix_date_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Исправление представления дат
    
    Требует наличия на первом месте столбца без даты
    
    :param df: Фрейм с некорректными данными
    :return: Исправленный датафрейм
    """
    new_columns = [df.columns[0]]
    for column in df.columns[1:]:
        column = float(column)
        date = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(column) - 2).strftime('%Y-%m-%d')
        new_columns.append(date)

    df.columns = new_columns
    return df


def load_file(
    file_path: str,
    header: int,
    need_data_fix: bool = False,
    engine: Union[Literal["xlrd", "openpyxl", "odf", "pyxlsb"], None] = "pyxlsb",
) -> pd.DataFrame:
    """
    Загрузка файла с данными

    :param file_path: Путь к файлу
    :param header: Номер строки, являющейся заголовком
    :param need_data_fix: Необходимость восстановления даты
    :param engine: Движок, который работает с файлом
    :return: Датафрейм с информацией из файла
    """
    df = pd.read_excel(
        io=file_path,
        engine=engine,
        header=header,  # У файла есть заголовок-объединение
    )
    if need_data_fix:
        df = fix_date_columns(df)
    return df


def plot_increasing(
    df: pd.DataFrame,
    feature: str,
    series: list,
    name: str,
) -> None:
    """
    Отрисовка графиков роста пользователей по периодам для конкретной переменной.

    График сохраняется в виде файла

    :param df: Фрейм с данными для отрисовки
    :param feature: Название переменной для отрисовки
    :param series: Временной ряд
    :param name: Название предмета, который исследуется
    :return: None
    """
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    fontsize = 18
    ax.tick_params(labelsize=fontsize) 
    if 'клиент' in feature:
        df[series].sum().plot(ax=ax)
        ax.set_title(f"Рост пользователей {name} за все периоды", {'fontsize': 18})
    else:
        new_df = df.groupby(feature)[series].sum().transpose()
        new_df.plot(ax=ax)
        ax.set_title(f"Рост пользователей {name} за все периоды по {feature}", {'fontsize': 18})
        ax.set_ylabel("Кол-во клиентов за период", {'fontsize': 18})
    # fig.savefig(f'{feature}.png')
    return fig


def plot_box(
    df: pd.DataFrame,
) -> None:
    """
    Отрисовка диаграммы "Ящик с усами"

    :param df: Фрейм данных для отображения
    :return: None
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    fontsize = 18
    plt.title('Ящик с усами распределения клиентов', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    sns.boxplot(
        data=df,
        x=df.index.year,
        y='y',
        ax=ax,
        boxprops=dict(alpha=.3),
    )
    sns.swarmplot(
        data=df,
        x=df.index.year,
        y='y',
        ax=ax,
    )
    ax.set_ylabel("Кол-во клиентов за период", {'fontsize': 18})
    ax.set_xlabel("Год", {'fontsize': 18})
    # sns.boxplot(data=df, x=df.index.year, y='y', ax=ax, boxprops=dict(alpha=.3))
    # sns.swarmplot(data=df, x=df.index.year, y='y')
    # fig.savefig(f'boxplot.png')
    return fig


def plot_chart(
    df: pd.DataFrame,
) -> None:
    """
    Отрисовка диаграммы изменения числа пользователей по годам

    :param df: Фрейм данных для отображения
    :return: None
    """
    # TODO Сохранение в файл
    fig, ax = plt.subplots(1, 1, figsize=(20, 28))
    alt.Chart(df.reset_index()).mark_line(point=True).encode(
        x='ds',
        y='y',
        column='year(ds)',
        tooltip=['ds', 'y']).properties(
        title="Рост клиентов за каждый год по отдельности",
        width=100,
    ).configure_header(
        titleColor='black',
        titleFontSize=18,
        labelColor='blue',
        labelFontSize=18,
    )
    # fig.savefig(f'chartplot.png')
    return fig


def plot_charTS(seria: pd.Series):
    """

    :param seria:
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 28))
    decompose = seasonal_decompose(seria)
    plt.title('Сезонная декомпозиция')
    plt.title('Сезонная декомпозиция')
    fig = decompose.plot()
    # fig.savefig(f'TS_plot.png')
    return fig


def main() -> None:
    """
    Запуск программы

    :return: None
    """
    print('Запуск')
    df_acquiring = load_file(file_path='dataset/acquiring.xlsb', header=1, need_data_fix=True)
    df_general = load_file(file_path='dataset/general.xlsb', header=0)

    tmp_cols = ['клиент', 'Сегмент id', 'Кластер', 'Средний возраст работников', 'Тип организации']
    df_acquiring_merged = pd.merge(df_general[tmp_cols], df_acquiring, on='клиент', how='outer')

    series_cols = df_acquiring.columns[df_acquiring.columns.str.contains("клиент") == False].values
    level_cols = df_acquiring.columns[df_acquiring.columns.str.contains("клиент")].values

    data = pd.DataFrame(series_cols, columns=["ds"])
    data['ds'] = pd.to_datetime(data['ds'], format='%Y-%m-%d')
    data.loc[:, "y"] = df_acquiring[series_cols].sum().values
    data = data.set_index('ds')

    print('Начало отрисовки графиков')
    for col in tmp_cols:
        plot_increasing(df=df_acquiring_merged, feature=col, series=series_cols, name='Эквайринг')

    # FIXME Смотри TODO по функции
    # plot_chart(data)

    plot_box(data)

    plot_charTS(data["y"])


if __name__ == "__main__":
    main()