import os
import requests
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from prefect import task, Flow
from prefect.schedules import IntervalSchedule


url_i = "https://api.carbonintensity.org.uk/intensity"
url_g = "https://api.carbonintensity.org.uk/generation"


def period_i(start, end):
    return requests.get(f"{url_i}/{start}/{end}").json()["data"]


def period_g(start, end):
    return requests.get(f"{url_g}/{start}/{end}").json()["data"]


def build_dfs(data_i, data_g):
    carbon_intensity = pd.DataFrame()
    carbon_intensity['timestamp'] = [pd.to_datetime(d['from'],
                                                    format='%Y-%m-%dT%H:%MZ')
                                     for d in data_i]
    carbon_intensity['intensity'] = [d['intensity']['actual'] for d in data_i]
    carbon_intensity['intensity_forecast'] = [d['intensity']['forecast'] for d
                                              in data_i]
    carbon_intensity.set_index('timestamp', inplace=True)

    energy_generation = pd.DataFrame()
    energy_generation['timestamp'] = [pd.to_datetime(d['from'],
                                                     format='%Y-%m-%dT%H:%MZ')
                                      for d in data_g]
    energy_generation['biomass'] = [d['generationmix'][0]['perc'] for d in
                                    data_g]
    energy_generation['coal'] = [d['generationmix'][1]['perc'] for d in
                                 data_g]
    energy_generation['imports'] = [d['generationmix'][2]['perc'] for d in
                                    data_g]
    energy_generation['gas'] = [d['generationmix'][3]['perc'] for d in
                                data_g]
    energy_generation['nuclear'] = [d['generationmix'][4]['perc'] for d in
                                    data_g]
    energy_generation['other'] = [d['generationmix'][5]['perc'] for d in
                                  data_g]
    energy_generation.set_index('timestamp', inplace=True)

    return pd.merge(carbon_intensity, energy_generation, how='inner',
                    left_index=True, right_index=True)


def aggregate_years():
    dfs = []
    for year in list(range(2018, 2023)):
        for month in list(range(1, 13)):
            data_i = period_i(start=f"{year}-{str(month).zfill(2)}-01T00:00Z",
                              end=f"{year}-{str(month).zfill(2)}-31T00:00Z")
            data_g = period_g(start=f"{year}-{str(month).zfill(2)}-01T00:00Z",
                              end=f"{year}-{str(month).zfill(2)}-31T00:00Z")
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

    print(f"Number of rows with missing values: "
          f"{df.isnull().any(axis=1).mean()}")

    return df


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
    return forecaster.predict(steps=48, last_window=y.intensity,
                              exog=y[['biomass', 'coal', 'imports', 'gas',
                                      'nuclear', 'other']])


@task(max_retries=300, retry_delay=timedelta(0, 3), log_stdout=True)
def extract():
    """
    Extract

    If failed this task will retry 300 times at 3 second interval and fail
    permenantly.
    """

    d = datetime.now()- timedelta(hours=48)
    data_i = period_i(start=f"{d.isoformat().split('.')[0]}Z",
                      end=f"{datetime.now().isoformat().split('.')[0]}Z")
    data_g = period_g(start=f"{d.isoformat().split('.')[0]}Z",
                      end=f"{datetime.now().isoformat().split('.')[0]}Z")
    return scrub_ts_data(build_dfs(data_i, data_g).drop(
        columns=['intensity_forecast']))


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
    from sqlalchemy import create_engine, Table, Column, Integer, String, \
        MetaData

    print(data)

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

        result = engine.execute(f"INSERT INTO forecasts (timestamp, "
                                f"prediction_date, prediction)\
            VALUES('{timestamp}', '{event_pred}', '{pred}') \
            ON CONFLICT (id) DO UPDATE SET timestamp = '{timestamp}', "
                                f"prediction = '{pred}';")

        result.close()

    return 0


if __name__ == '__main__':
    URI = sys.argv[0]

    if URI is None:
        URI = f"sqlite:///working/intensity_forecasts.db"

    # Create a schedule object.
    # This object starts from the time of script execution and repeats once
    # every 30 minutes.
    schedule = IntervalSchedule(
        start_date=datetime.utcnow(),
        interval=timedelta(minutes=30),
    )

    # Attach the schedule object and orchastrate flow.
    with Flow("Predict Half-Hour Carbon Intensities",
              schedule=schedule) as flow:
        e = extract()
        t = transform(e, f"{os.path.dirname(__file__)}/models/elastic_net.pkl")
        l = load(t, URI)

    flow.run()
