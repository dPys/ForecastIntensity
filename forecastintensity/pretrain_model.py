import os
import requests
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from sklearn.linear_model import ElasticNet
from prefect import task, Flow
from prefect.schedules import IntervalSchedule
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.model_selection import grid_search_forecaster, \
    backtesting_forecaster
from . import period_i, period_g, build_dfs, scrub_ts_data, aggregate_years,

url_i = "https://api.carbonintensity.org.uk/intensity"
url_g = "https://api.carbonintensity.org.uk/generation"


def train_model(df):

    # Split data into train-val-test
    # =========================================================================
    data = df.loc['2018-05-10 00:00:00': '2022-05-27 00:00:00']
    end_train = '2020-5-31 00:00:00'
    end_validation = '2022-4-29 00:00:00'
    data_train = data.loc[: end_train, :]
    data_val   = data.loc[end_train:end_validation, :]
    data_test  = data.loc[end_validation:, :]

    fig, ax = plt.subplots(figsize=(12, 4))
    data_train.intensity.plot(ax=ax, label='train', linewidth=1)
    data_val.intensity.plot(ax=ax, label='val', linewidth=1)
    data_test.intensity.plot(ax=ax, label='test', linewidth=1)
    ax.set_title('Carbon Intensity')
    ax.legend();

    zoom = ('2021-01-01 00:00:00','2021-12-31 00:00:00')
    fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)
    main_ax = fig.add_subplot(grid[1:3, :])
    zoom_ax = fig.add_subplot(grid[5:, :])
    data.intensity.plot(ax=main_ax, c='black', alpha=0.5, linewidth=0.5)
    min_y = min(data.intensity)
    max_y = max(data.intensity)
    main_ax.fill_between(zoom, min_y, max_y, facecolor='blue', alpha=0.5,
                         zorder=0)
    main_ax.set_xlabel('')
    data.loc[zoom[0]: zoom[1]].intensity.plot(ax=zoom_ax, color='blue',
                                              linewidth=2)
    main_ax.set_title(f'Carbon intensity: {data.index.min()}, '
                      f'{data.index.max()}', fontsize=14)
    zoom_ax.set_title(f'Carbon intensity: {zoom}', fontsize=14)
    plt.subplots_adjust(hspace=1)
    plt.show()

    forecaster = ForecasterAutoreg(
                regressor = make_pipeline(StandardScaler(),
                                          ElasticNet(random_state=42)),
                lags      = 48
            )

    param_grid = {'elasticnet__alpha': [1e-2, 1e-1, 0.25, 0.5, 1],
                  'elasticnet__l1_ratio': [0, 0.25, 0.50, 0.75, 1]}
    lags_grid = [24, 36, 48, 72]

    results_grid = grid_search_forecaster(
                            forecaster  = forecaster,
                            y           = data.loc[:end_validation,
                                          'intensity'],
                            exog        = data.loc[:end_validation,
                                          ['biomass', 'coal', 'imports',
                                           'gas', 'nuclear', 'other']],
                            param_grid  = param_grid,
                            lags_grid   = lags_grid,
                            steps       = 48,
                            metric      = 'mean_absolute_error',
                            initial_train_size = len(data[:end_train]),
                            return_best = True,
                            verbose     = False
                      )

    forecaster = ForecasterAutoreg(
                regressor = make_pipeline(StandardScaler(),
                                          ElasticNet(alpha=0.1, l1_ratio=0.75,
                                                     random_state=42)),
                lags      = list(range(1,49))
            )
    forecaster.fit(y=data[:end_validation].intensity,
                   exog=data[:end_validation][['biomass', 'coal', 'imports',
                                               'gas', 'nuclear', 'other']])

    # Backtest
    # =====================================================================
    metric, predictions = backtesting_forecaster(
                                forecaster = forecaster,
                                y          = data.intensity,
                                exog       = data[['biomass', 'coal',
                                                   'imports', 'gas', 'nuclear',
                                                   'other']],
                                initial_train_size = len(
                                    data.loc[:end_validation]),
                                steps      = 48,
                                metric     = 'mean_absolute_error',
                                verbose    = True
                            )

    predictions = pd.Series(data=predictions.pred,
                            index=data[end_validation:].index)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    data.loc[predictions.index, 'intensity'].plot(ax=ax, linewidth=2,
                                                  label='real')
    predictions.plot(linewidth=2, label='prediction', ax=ax)
    ax.set_title('Forecast vs Real Intensity')
    ax.legend();
    plt.show()

    print(f'Backtest error: {metric}')

    joblib.dump(forecaster, f"./models/elastic_net.pkl")

    return 0


if __name__ == '__main__':
    df = aggregate_years()
    df.to_csv(f"./data/training_data.csv")
    df = scrub_ts_data(df)
    df.to_csv(f"./data/training_data_scrubbed.csv")
    out = train_model(df)
