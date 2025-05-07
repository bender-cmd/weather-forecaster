# file: assessment/forecast.py
import pandas as pd
import argparse
from prophet import Prophet


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return weather dataset from a CSV file."""
    df = pd.read_csv(filepath)
    df['ds'] = pd.to_datetime(df['ds'])
    return df


def train_model(df: pd.DataFrame) -> Prophet:
    """Train a basic Prophet model. Extend with tuning if needed."""
    model = Prophet()

    # TODO: Add external regressor  humidity
    # TODO: Replace with tuned Prophet model using Optuna

    model.fit(df)
    return model


def make_future_dataframe(model: Prophet, df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Create future dataframe for forecasting."""
    future = model.make_future_dataframe(periods=periods)

    # TODO: Add regressor values to future dataframe if used

    return future


def generate_forecast(model: Prophet, future: pd.DataFrame) -> pd.DataFrame:
    """Generate forecast using the model."""
    forecast = model.predict(future)
    return forecast


def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame):
    """Visualize the forecast vs. actuals.

    TODO: Implement this function to plot results using matplotlib or plotly
    """
    pass


def main():
    parser = argparse.ArgumentParser(description="Weather Forecast using Prophet")

    # TODO: Add CLI arguments:
    # --input: path to input CSV file
    # --periods: forecast horizon
    # --output: path to output CSV
    # --use_regressor: flag to use humidity
    # parser.add_argument('--input', ...)
    args = parser.parse_args()

    df = load_data(args.input)

    # Rename the target variable for Prophet
    df.rename(columns={'temperature_celsius': 'y'}, inplace=True)

    model = train_model(df)
    future = make_future_dataframe(model, df, args.periods)
    forecast = generate_forecast(model, future)

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # TODO: Call plot_forecast and save CSV if output provided
    # plot_forecast(df, forecast)
    # forecast.to_csv(args.output)


if __name__ == "__main__":
    main()
