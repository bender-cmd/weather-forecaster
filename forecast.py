#!/usr/bin/env python3
"""
Weather Temperature Forecasting with Prophet.

Usage:
    forecast.py --input <file.csv> [--output <forecast.csv>] [--periods <days>]
                [--use-regressor [<regressor>...]] [--plot-components]
                [--no-tuning] [--trials <n>]
    forecast.py (-h | --help)

Options:
    --input <file.csv>          Path to input CSV (required).
    --output <forecast.csv>     Save forecast to this file [default: None].
    --periods <days>           Days to forecast [default: 7].
    --use-regressor <regressor> Regressors to use [default: humidity, pressure].
    --plot-components           Show component plots.
    --no-tuning                Skip hyperparameter tuning.
    --trials <n>               Number of hyperparameter trials [default: 30].
    -h --help                  Show this help.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class WeatherForecaster:
    """A class for weather temperature forecasting using Prophet."""

    def __init__(self) -> None:
        """Initialize the forecaster with default settings."""
        self.model: Optional[Prophet] = None
        self.forecast: Optional[pd.DataFrame] = None
        self.training_data: Optional[pd.DataFrame] = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess weather dataset from a CSV file.

        Args:
            filepath: Path to the input CSV file.

        Returns:
            Cleaned DataFrame with datetime index and temperature column.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            FileNotFoundError: If input file doesn't exist.
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Input file not found: {filepath}")

            # Read with optimized parameters
            df = pd.read_csv(
                filepath,
                usecols=["ds", "temperature_celsius"],  # Only read needed columns
                parse_dates=["ds"],
                low_memory=False,
            )

            # Validate data
            if len(df) < 30:
                raise ValueError("Insufficient data - need at least 30 observations")
            if df["temperature_celsius"].isna().sum() > len(df) * 0.1:  # 10% threshold
                raise ValueError("Too many missing temperature values")

            # Optimize memory usage
            df["temperature_celsius"] = pd.to_numeric(
                df["temperature_celsius"], downcast="float"
            )
            df.rename(columns={"temperature_celsius": "y"}, inplace=True)

            df.rename(columns={'temperature_celsius': 'y'}, inplace=True)
            return df

        except pd.errors.EmptyDataError as e:
            raise ValueError("Input file is empty") from e
        except pd.errors.ParserError as e:
            raise ValueError("Invalid CSV format") from e

    def _instantiate_model(self, params: Optional[Dict] = None) -> Prophet:
        """
        Create a Prophet model with specified parameters.

        Args:
            params: Dictionary of Prophet parameters to override defaults.

        Returns:
            Configured Prophet model instance.
        """
        default_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,  # Controls how strongly seasonal patterns influence forecasts
            'seasonality_mode': 'additive',  # Seasonality effects are added to trend (constant magnitude)
            # Auto - detects patterns at these frequencies
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True
        }

        if params:
            default_params.update(params)

        model = Prophet(**default_params)
        model.add_country_holidays(country_name='ZA')  # leverage south african holidays for weather forecasting
        return model

    def _should_use_regressor(self, df: pd.DataFrame, regressor_col: str) -> bool:
        """
        Determine if a regressor should be included based on data quality and correlation.

        Args:
            df: DataFrame containing the data.
            regressor_col: Column name to check as potential regressor.

        Returns:
            bool: True if regressor should be used, False otherwise.
        """
        if regressor_col not in df.columns:
            logger.warning("Regressor %s not in data", regressor_col)
            return False

        # Check for sufficient non-null values
        if df[regressor_col].isna().mean() > 0.2:  # 20% missing threshold
            logger.warning(
                "Regressor %s has too many missing values (%.1f%%)",
                regressor_col,
                df[regressor_col].isna().mean() * 100,
            )
            return False

        # Check correlation
        corr = df[[regressor_col, "y"]].corr().loc[regressor_col, "y"]
        if abs(corr) < 0.3:  # Moderate correlation threshold
            logger.warning(
                "Regressor '%s' ignored due to weak correlation (r=%.2f)",
                regressor_col,
                corr,
            )
            return False

        return True

    def _early_stopping_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Stop optimization early if results aren't improving."""
        if trial.number > 100 and study.best_value > 2.0:
            raise optuna.exceptions.TrialPruned()

    def _tune_hyperparameters(
        self, df: pd.DataFrame, n_trials: int = 30
    ) -> Dict[str, float]:
        """
        Optimize Prophet hyperparameters using Optuna.

        Args:
            df: Training data.
            n_trials: Number of optimization trials.

        Returns:
            Dictionary of optimized parameters.
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                "changepoint_prior_scale": trial.suggest_float(
                    "changepoint_prior_scale", 0.001, 0.5, log=True
                ),
                "seasonality_prior_scale": trial.suggest_float(
                    "seasonality_prior_scale", 0.1, 20.0
                ),
                # testing both modes to determine which
                # fits the temperature data better
                "seasonality_mode": trial.suggest_categorical(
                    "seasonality_mode", ["additive", "multiplicative"]
                ),
                # controls how strongly holidays affect temperatures
                "holidays_prior_scale": trial.suggest_float(
                    "holidays_prior_scale", 0.1, 10.0
                ),
            }

            model = self._instantiate_model(params)
            model.fit(df)

            # Dynamic cross-validation parameters
            min_date, max_date = df["ds"].min(), df["ds"].max()
            total_days = (max_date - min_date).days

            initial = "300 days" if total_days <= 366 else "365 days"
            period = "30 days" if total_days <= 366 else "90 days"
            horizon = "30 days"

            df_cv = cross_validation(
                model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes",
            )

            return performance_metrics(df_cv)["rmse"].mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=min(10, n_trials // 3),  # 1/3 of trials for exploration
                multivariate=True,
            ),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[self._early_stopping_callback],
            n_jobs=-1,  # Use all available cores
        )
        logger.info("Best trial: %s", study.best_trial.number)
        logger.info("Best RMSE: %.4f", study.best_value)
        return study.best_params

    def train(
        self,
        df: pd.DataFrame,
        use_regressor: Optional[List[str]] = None,
        tune_hyperparameters: bool = True,
        n_trials: int = 30,
    ) -> None:
        """
        Train the forecasting model.

        Args:
            df: Training data.
            use_regressor: List of columns to consider as regressors.
            tune_hyperparameters: Whether to optimize model parameters.
            n_trials: Number of hyperparameter optimization trials.

        Raises:
            ValueError: If training data is insufficient.
        """
        logger.info("Starting model training with %d data points", len(df))
        self.training_data = df.copy()

        # Hyperparameter tuning
        best_params = (
            self._tune_hyperparameters(df, n_trials) if tune_hyperparameters else None
        )
        self.model = self._instantiate_model(best_params)

        # Add validated regressors
        if use_regressor is not None:
            valid_regressors = [
                col
                for col in use_regressor
                if self._should_use_regressor(df, col)
            ][:3]  # Limit to top 3 regressors

            for col in valid_regressors:
                self.model.add_regressor(col)
                logger.info("Added regressor: %s", col)

        self.model.fit(df)
        logger.info("Model training completed")

    def predict(self, periods: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate forecast for future periods.

        Args:
            periods: Number of future periods to forecast.

        Returns:
            Tuple of (future dataframe, forecast dataframe).

        Raises:
            ValueError: If model hasn't been trained.
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        logger.info("Generating %d-day forecast", periods)
        future = self.model.make_future_dataframe(periods=periods)

        # Handle regressors if present
        if hasattr(self.model, "extra_regressors"):
            for regressor in self.model.extra_regressors:
                if regressor in self.training_data.columns:
                    future = future.merge(
                        self.training_data[["ds", regressor]],
                        on="ds",
                        how="left",
                    )
                    future[regressor] = future[regressor].ffill()

        self.forecast = self.model.predict(future)
        return future, self.forecast

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance on training data.

        Returns:
            Dictionary of evaluation metrics.

        Raises:
            ValueError: If forecast or training data isn't available.
        """
        if self.forecast is None or self.training_data is None:
            raise ValueError("Model must be trained and forecast generated")

        merged = pd.merge(
            self.training_data[["ds", "y"]],
            self.forecast[["ds", "yhat"]],
            on="ds",
        )

        y_true = merged["y"]
        y_pred = merged["yhat"]

        return {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }

    def plot_forecast(
        self, plot_components: bool = False, last_n_days: int = 7
    ) -> None:
        """
        Visualize the forecast results.

        Args:
            plot_components: Whether to show trend/seasonality components.
            last_n_days: Number of most recent days to highlight.

        Raises:
            ValueError: If forecast isn't available.
        """
        if self.forecast is None or self.training_data is None:
            raise ValueError("No forecast available to plot")

        # Main forecast plot
        plt.figure(figsize=(12, 6))
        forecast_subset = self.forecast.iloc[-last_n_days:]

        plt.plot(
            forecast_subset["ds"],
            forecast_subset["yhat"],
            label="Forecast",
            color="blue",
            marker="o",
        )
        plt.fill_between(
            forecast_subset["ds"],
            forecast_subset["yhat_lower"],
            forecast_subset["yhat_upper"],
            color="blue",
            alpha=0.2,
            label="Confidence Interval",
        )
        plt.title(f"{last_n_days}-Day Temperature Forecast")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Actual vs forecast plot
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.training_data["ds"],
            self.training_data["y"],
            label="Actual",
            color="black",
        )
        plt.plot(
            self.forecast["ds"],
            self.forecast["yhat"],
            label="Forecast",
            color="blue",
        )
        plt.fill_between(
            self.forecast["ds"],
            self.forecast["yhat_lower"],
            self.forecast["yhat_upper"],
            color="blue",
            alpha=0.2,
        )
        plt.title("Actual vs Forecast Temperatures")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Recent history + forecast
        if last_n_days and len(self.forecast) > last_n_days:
            plt.figure(figsize=(12, 6))
            cutoff_date = self.forecast["ds"].iloc[-1] - pd.Timedelta(days=2 * last_n_days)

            actu_subset = self.training_data[self.training_data["ds"] >= cutoff_date]
            fcst_subset = self.forecast[self.forecast["ds"] >= cutoff_date]

            merged = pd.merge(fcst_subset[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], actu_subset[['ds', 'y']], on='ds',
                              how='left')
            merged.rename(columns={'y': 'actual', 'yhat': 'forecast', 'ds': 'date'}, inplace=True)
            merged['diff'] = merged['forecast'] - merged['actual']
            print(merged.tail(100))

            plt.plot(
                actu_subset["ds"],
                actu_subset["y"],
                label="Actual",
                color="black",
                marker="o",
            )
            plt.plot(
                fcst_subset["ds"],
                fcst_subset["yhat"],
                label="Forecast",
                color="blue",
                marker="o",
            )
            plt.fill_between(
                fcst_subset["ds"],
                fcst_subset["yhat_lower"],
                fcst_subset["yhat_upper"],
                color="blue",
                alpha=0.2,
            )
            plt.title(f"Last {2 * last_n_days} Days + Forecast")
            plt.xlabel("Date")
            plt.ylabel("Temperature (°C)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        if plot_components:
            self.model.plot_components(self.forecast)
            plt.show()


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weather Temperature Forecasting with Prophet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with columns: ds, temperature_celsius",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save forecast results (CSV)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=7,
        help="Number of days to forecast",
    )
    parser.add_argument(
        "--use-regressor",
        nargs="*",
        default=None,
        help="Columns to use as additional regressors",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip hyperparameter tuning",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of hyperparameter optimization trials",
    )
    parser.add_argument(
        "--plot-components",
        action="store_true",
        help="Show component plots",
    )

    return parser.parse_args()


def save_forecast_to_csv(forecast: pd.DataFrame, output_path: Path, periods: int) -> None:
    """Save forecast results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast[["ds", "yhat"]].tail(periods).to_csv(output_path, index=False)
    logger.info("Forecast saved to %s", output_path.resolve())


def main() -> int:
    """Main execution function."""
    args = parse_arguments()

    try:
        forecaster = WeatherForecaster()
        df = forecaster.load_data(args.input)

        # Train model
        forecaster.train(
            df,
            use_regressor=args.use_regressor,
            tune_hyperparameters=not args.no_tuning,
            n_trials=args.trials,
        )

        # Generate forecast
        _, forecast = forecaster.predict(periods=args.periods)

        # Print evaluation metrics
        metrics = forecaster.evaluate()
        logger.info("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            logger.info("%s: %.2f", metric.upper(), value)

        # Visualize results
        forecaster.plot_forecast(
            plot_components=args.plot_components,
            last_n_days=args.periods,
        )

        # Save results if requested
        if args.output:
            save_forecast_to_csv(forecast, Path(args.output), args.periods)

        return 0

    except Exception as e:
        logger.error("Error: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())