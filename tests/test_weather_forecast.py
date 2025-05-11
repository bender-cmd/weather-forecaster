#!/usr/bin/env python3

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forecast import WeatherForecaster

@pytest.fixture
def sample_df():
    # Generate dummy weather data for 60 days
    dates = pd.date_range(start='2024-01-01', periods=60)
    data = {
        'ds': dates,
        'y': 20 + (dates.dayofyear % 10),  # seasonal-like pattern
        'humidity': 50 + (dates.dayofyear % 5),
        'pressure': 1010 + (dates.dayofyear % 3)
    }
    return pd.DataFrame(data)


def test_load_data(tmp_path):
    # Create a sample CSV
    data = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=10),
        'temperature_celsius': range(10)
    })
    file = tmp_path / "sample.csv"
    data.to_csv(file, index=False)

    forecaster = WeatherForecaster()
    df = forecaster.load_data(str(file))

    assert 'ds' in df.columns
    assert 'y' in df.columns
    assert len(df) == 10  # why 10?
    assert pd.api.types.is_datetime64_any_dtype(df['ds'])


def test_train_model(sample_df):
    forecaster = WeatherForecaster()
    forecaster.train(sample_df, tune_hyperparameters=False)

    assert forecaster.model is not None
    assert forecaster.training_data is not None


def test_forecast_output_shape(sample_df):
    forecaster = WeatherForecaster()
    forecaster.train(sample_df, tune_hyperparameters=False)
    future, forecast = forecaster.predict(periods=5)

    assert len(forecast) >= len(sample_df) + 5
    assert {'ds', 'yhat'}.issubset(forecast.columns)


def test_evaluate_metrics(sample_df):
    forecaster = WeatherForecaster()
    forecaster.train(sample_df, tune_hyperparameters=False)
    forecaster.predict(periods=5)
    metrics = forecaster.evaluate()

    assert all(metric in metrics for metric in ['mse', 'mae', 'rmse'])
    assert all(isinstance(value, float) for value in metrics.values())


def test_use_invalid_regressor_warning(sample_df):
    forecaster = WeatherForecaster()
    with pytest.warns(UserWarning, match="Regressor invalid_regressor not in data"):
        forecaster.train(sample_df, use_regressor=['invalid_regressor'], tune_hyperparameters=False)


def test_should_use_regressor_valid(sample_df):
    forecaster = WeatherForecaster()
    assert forecaster._should_use_regressor(sample_df, 'humidity') is True


def test_should_use_regressor_invalid(sample_df):
    forecaster = WeatherForecaster()
    assert forecaster._should_use_regressor(sample_df, 'non_existent_column') is False
