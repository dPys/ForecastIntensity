import os
import requests
import json
import joblib
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from sklearn.linear_model import ElasticNet
from prefect import task, Flow
from prefect.schedules import IntervalSchedule
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster


url_i = "https://api.carbonintensity.org.uk/intensity"
url_g = "https://api.carbonintensity.org.uk/generation"

model_dir = './'
output_dir = './'

def period_i(start, end):
    return requests.get(f"{url_i}/{start}/{end}").json()["data"]


def period_g(start, end):
    return requests.get(f"{url_g}/{start}/{end}").json()["data"]


def build_dfs(data_i, data_g):
    carbon_intensity = pd.DataFrame()
    carbon_intensity['timestamp'] = [pd.to_datetime(d['from'], format='%Y-%m-%dT%H:%MZ') for d in data_i]
    carbon_intensity['intensity'] = [d['intensity']['actual'] for d in data_i]
    carbon_intensity['intensity_forecast'] = [d['intensity']['forecast'] for d in data_i]
    carbon_intensity.set_index('timestamp', inplace=True)

    energy_generation = pd.DataFrame()
    energy_generation['timestamp'] = [pd.to_datetime(d['from'], format='%Y-%m-%dT%H:%MZ') for d in data_g]
    energy_generation['biomass'] = [d['generationmix'][0]['perc'] for d in data_g]
    energy_generation['coal'] = [d['generationmix'][1]['perc'] for d in data_g]
    energy_generation['imports'] = [d['generationmix'][2]['perc'] for d in data_g]
    energy_generation['gas'] = [d['generationmix'][3]['perc'] for d in data_g]
    energy_generation['nuclear'] = [d['generationmix'][4]['perc'] for d in data_g]
    energy_generation['other'] = [d['generationmix'][5]['perc'] for d in data_g]
    energy_generation.set_index('timestamp', inplace=True)

    return pd.merge(carbon_intensity, energy_generation, how='inner', left_index=True, right_index=True)


def aggregate_years():
    dfs = []
    for year in list(range(2018, 2023)):
        for month in list(range(1, 13)):
            data_i = period_i(start=f"{year}-{str(month).zfill(2)}-01T00:00Z", end=f"{year}-{str(month).zfill(2)}-31T00:00Z")
            data_g = period_g(start=f"{year}-{str(month).zfill(2)}-01T00:00Z", end=f"{year}-{str(month).zfill(2)}-31T00:00Z")
            dfs.append(build_dfs(data_i, data_g))
    return pd.concat(dfs)


def scrub_ts_data(df):
    df = df[~df.index.duplicated()]
    df = df.dropna()
    df = df.asfreq(freq='30min', fill_value=np.nan)
    df = df.sort_index()
    df = df.fillna(df.ewm(span=48).mean())

    # (df.index == pd.date_range(start=df.index.min(),
    #                            end=df.index.max(),
    #                            freq=df.index.freq)).all()

    print(f"Number of rows with missing values: {df.isnull().any(axis=1).mean()}")

    return df


# def train_model(df, model_dir):
#
#     # Split data into train-val-test
#     # ==============================================================================
#     data = df.loc['2018-05-10 00:00:00': '2022-05-27 00:00:00']
#     end_train = '2020-5-31 00:00:00'
#     end_validation = '2022-4-29 00:00:00'
#     data_train = data.loc[: end_train, :]
#     data_val   = data.loc[end_train:end_validation, :]
#     data_test  = data.loc[end_validation:, :]
#
#     # import matplotlib.pyplot as plt
#     # fig, ax = plt.subplots(figsize=(12, 4))
#     # data_train.intensity.plot(ax=ax, label='train', linewidth=1)
#     # data_val.intensity.plot(ax=ax, label='val', linewidth=1)
#     # data_test.intensity.plot(ax=ax, label='test', linewidth=1)
#     # ax.set_title('Carbon Intensity')
#     # ax.legend();
#
#     # zoom = ('2021-01-01 00:00:00','2021-12-31 00:00:00') #Our zoom period is from
#     # fig = plt.figure(figsize=(12, 6))
#     # grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)
#     # main_ax = fig.add_subplot(grid[1:3, :])
#     # zoom_ax = fig.add_subplot(grid[5:, :])
#     # data.intensity.plot(ax=main_ax, c='black', alpha=0.5, linewidth=0.5)
#     # min_y = min(data.intensity)
#     # max_y = max(data.intensity)
#     # main_ax.fill_between(zoom, min_y, max_y, facecolor='blue', alpha=0.5, zorder=0)
#     # main_ax.set_xlabel('')
#     # data.loc[zoom[0]: zoom[1]].intensity.plot(ax=zoom_ax, color='blue', linewidth=2)
#     # main_ax.set_title(f'Carbon intensity: {data.index.min()}, {data.index.max()}', fontsize=14)
#     # zoom_ax.set_title(f'Carbon intensity: {zoom}', fontsize=14)
#     # plt.subplots_adjust(hspace=1)
#     # plt.show()
#
#     forecaster = ForecasterAutoreg(
#                 regressor = make_pipeline(StandardScaler(), ElasticNet(random_state=42)),
#                 lags      = 48
#             )
#
#     param_grid = {'elasticnet__alpha': [1e-3, 1e-2, 1e-1, 0.25, 0.5, 1], 'elasticnet__l1_ratio': [0, 0.25, 0.50, 0.75, 1]}
#     lags_grid = [24, 36, 48, 72]
#
#     results_grid = grid_search_forecaster(
#                             forecaster  = forecaster,
#                             y           = data.loc[:end_validation, 'intensity'],
#                             exog        = data.loc[:end_validation, ['biomass', 'coal', 'imports', 'gas', 'nuclear', 'other']],
#                             param_grid  = param_grid,
#                             lags_grid   = lags_grid,
#                             steps       = 48,
#                             metric      = 'mean_absolute_error',
#                             initial_train_size = len(data[:end_train]),
#                             return_best = True,
#                             verbose     = False
#                       )
#
#     forecaster = ForecasterAutoreg(
#                 regressor = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.75, random_state=42)),
#                 lags      = list(range(1,49))
#             )
#     forecaster.fit(y=data[:end_validation].intensity, exog=data[:end_validation][['biomass', 'coal', 'imports', 'gas', 'nuclear', 'other']])
#
#     # Backtest
#     # =====================================================================
#     metric, predictions = backtesting_forecaster(
#                                 forecaster = forecaster,
#                                 y          = data.intensity,
#                                 exog       = data[['biomass', 'coal', 'imports', 'gas', 'nuclear', 'other']],
#                                 initial_train_size = len(data.loc[:end_validation]),
#                                 steps      = 48,
#                                 metric     = 'mean_absolute_error',
#                                 verbose    = True
#                             )
#
#     predictions = pd.Series(data=predictions.pred, index=data[end_validation:].index)
#
#     fig, ax = plt.subplots(figsize=(12, 3.5))
#     data.loc[predictions.index, 'intensity'].plot(ax=ax, linewidth=2, label='real')
#     predictions.plot(linewidth=2, label='prediction', ax=ax)
#     ax.set_title('Forecast vs Real Intensity')
#     ax.legend();
#     plt.show()
#
#     print(f'Backtest error: {metric}')
#
#     joblib.dump(forecaster, f"{model_dir}/reg.pkl")
#
#     return 0
#
#
# df = aggregate_years()
# df.to_csv(f"{model_dir}/training_data.csv")
# df = scrub_ts_data(df)
# df.to_csv(f"{model_dir}/training_data_scrubbed.csv")
# out = train_model(df, model_dir)


############ Serve forecaster every 30 minutes ##############
def backtest_predict_next_24h(forecaster, y):

    '''
    Backtest ForecasterAutoreg object when predicting next 48 30-minute
    intervals for the subsequent 24 hours

    Parameters
    ----------
    forecaster : ForecasterAutoreg
        ForecasterAutoreg object already trained.

    y : pd.Series with datetime index sorted
        Test time series values.

    Returns
    -------
    predictions: pd.Series
        Value of predictions.

    '''
    return forecaster.predict(steps=48, last_window=y.intensity, exog=y[['biomass', 'coal', 'imports', 'gas', 'nuclear', 'other']])


@task(max_retries=300, retry_delay=timedelta(0, 3), log_stdout=True)
def extract():
    """
    Extract

    If failed this task will retry 300 times at 3 second interval and fail permenantly.
    """

    d = datetime.now()- timedelta(hours=48)
    data_i = period_i(start=f"{d.isoformat().split('.')[0]}Z", end=f"{datetime.now().isoformat().split('.')[0]}Z")
    data_g = period_g(start=f"{d.isoformat().split('.')[0]}Z", end=f"{datetime.now().isoformat().split('.')[0]}Z")
    return scrub_ts_data(build_dfs(data_i, data_g).drop(columns=['intensity_forecast']))


@task(log_stdout=True)
def transform(x, model_path):
    """
    Extract
    """
    forecaster = joblib.load(model_path)
    return backtest_predict_next_24h(forecaster, x)


@task(max_retries=10, retry_delay=timedelta(0, 1), log_stdout=True)
def load(data, uri:str):
    """
    This task will save the prediction to an output file.
    If failed, this task will retry for 10 times and then fail permenantly.
    """
    from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData

    engine = create_engine(uri)

    if not engine.dialect.has_table(engine, 'forecasts'):
        meta = MetaData()
        forecasts = Table(
           'forecasts', meta,
           Column('id', Integer, primary_key = True),
           Column('timestamp', String),
           Column('prediction_date', String),
           Column('prediction', String),
        )
        meta.create_all(engine)

    query.close()

    dates = pd.date_range(
        start = data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
        periods = 48,
        freq = '30min'
        )

    upload_data = list(zip([
    datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * (num_predictions),
    [event.strftime('%Y-%m-%d %H:%M:%S') for event in dates ],
        predictions[1:]
    ))

    # Insert the data
    for upload_event in upload_data:
        timestamp, event_pred, pred = upload_event
        pred = round(pred, 4)

        result = engine.execute(f"INSERT INTO forecasts (timestamp, prediction_date, prediction)\
            VALUES('{timestamp}', '{event_pred}', '{pred}') \
            ON CONFLICT (id) DO UPDATE SET timestamp = '{timestamp}', prediction = '{pred}';")

        result.close()

    return 0


# Create a schedule object.
# This object starts from the time of script execution and repeats once every 30 minutes.
schedule = IntervalSchedule(
    start_date=datetime.utcnow(),
    interval=timedelta(minutes=30),
)


# Attach the schedule object and orchastrate flow.
with Flow("Predict Half-Hour Carbon Intensities", schedule=schedule) as flow:
    e = extract()
    t = transform(e, f"{model_dir}/reg.pkl")
    l = load(t, f"sqlite:///{output_dir}/intensity_forecasts.db")


if __name__ == '__main__':
    flow.run()
