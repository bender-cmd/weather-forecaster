#!/usr/bin/env python3
"""Temperature forecasting using Prophet."""

import argparse
import warnings
from typing import Optional, Dict, Tuple, List
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import optuna


class WeatherForecaster:
    """A class for weather temperature forecasting using Prophet."""

    def __init__(self):
        """Initialize the forecaster with default settings."""
        self.model = None
        self.forecast = None
        self.training_data = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess weather dataset from a CSV file.

        Args:
            filepath: Path to the input CSV file

        Returns:
            Cleaned DataFrame with datetime index and temperature column
        """
        try:
            df = pd.read_csv(filepath)
            df['ds'] = pd.to_datetime(df['ds'])

            # Basic data validation
            if 'temperature_celsius' not in df.columns:
                raise ValueError("Input data must contain 'temperature_celsius' column")

            df.rename(columns={'temperature_celsius': 'y'}, inplace=True)
            return df

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}") from e

    def _instantiate_model(self, params: Optional[Dict] = None) -> Prophet:
        """
        Create a Prophet model with specified parameters.

        Args:
            params: Dictionary of Prophet parameters

        Returns:
            Configured Prophet model instance
        """
        default_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True
        }

        if params:
            default_params.update(params)

        return Prophet(**default_params)

    def _should_use_regressor(self, df: pd.DataFrame, regressor_col: str) -> bool:
        """
        Determine if a regressor should be included based on correlation.

        Args:
            df: DataFrame containing the data
            regressor_col: Column name to check as potential regressor

        Returns:
            Boolean indicating whether to use the regressor
        """
        if regressor_col not in df.columns:
            warnings.warn(
                f"Regressor {regressor_col} not in data",
                UserWarning
            )
            return False

        corr = df[[regressor_col, 'y']].corr().loc[regressor_col, 'y']
        if abs(corr) < 0.3:  # Moderate correlation threshold
            warnings.warn(
                f"Regressor '{regressor_col}' ignored due to weak correlation (r={corr:.2f})",
                UserWarning
            )
            return False
        return True

    def _tune_hyperparameters(self, df: pd.DataFrame, n_trials: int = 30) -> Dict:
        """
        Optimize Prophet hyperparameters using Optuna.

        Args:
            df: Training data
            n_trials: Number of optimization trials

        Returns:
            Dictionary of optimized parameters
        """

        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float(
                    'changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float(
                    'seasonality_prior_scale', 0.1, 20.0),
                'seasonality_mode': trial.suggest_categorical(
                    'seasonality_mode', ['additive', 'multiplicative']),
                'holidays_prior_scale': trial.suggest_float(
                    'holidays_prior_scale', 0.1, 10.0)
            }

            model = self._instantiate_model(params)
            model.fit(df)

            # Cross-validation within the training set
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)

            merged = pd.merge(df[['ds', 'y']], forecast[['ds', 'yhat']], on='ds')
            return mean_squared_error(merged['y'], merged['yhat'])

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params

    def train(
            self,
            df: pd.DataFrame,
            use_regressor: Optional[List[str]] = None,
            tune_hyperparameters: bool = True,
            n_trials: int = 30
    ) -> None:
        """
        Train the forecasting model.

        Args:
            df: Training data
            use_regressor: List of columns to consider as regressors. If flag is active but no regressor defined, defaults will be tried
            tune_hyperparameters: Whether to optimize model parameters
            n_trials: Number of hyperparameter optimization trials

        Raises:
            ValueError: If training data is insufficient
        """
        if len(df) < 30:
            raise ValueError("Insufficient data - need at least 30 observations")

        self.training_data = df.copy()

        # Hyperparameter tuning
        best_params = self._tune_hyperparameters(df) if tune_hyperparameters else None

        # Initialize model with best parameters
        self.model = self._instantiate_model(best_params)

        # Add regressors if requested and valid
        if isinstance(use_regressor, list):  # if regressors requested
            if len(use_regressor) > 0:  # if regressors defined
                for regressor_col in use_regressor:
                    if self._should_use_regressor(df, regressor_col):
                        self.model.add_regressor(regressor_col)
            else:  # try defaults
                if self._should_use_regressor(df, 'humidity'):
                    self.model.add_regressor('humidity')
                if self._should_use_regressor(df, 'pressure'):
                    self.model.add_regressor('pressure')

        # Fit the model
        self.model.fit(df)

    def predict(self, periods: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate forecast for future periods.

        Args:
            periods: Number of future periods to forecast

        Returns:
            Tuple of (future dataframe, forecast dataframe)
        """
        if not self.model:
            raise ValueError("Model must be trained before making predictions")

        # Create future dataframe with regressors if they exist
        future = self.model.make_future_dataframe(periods=periods)

        # Handle regressors by forward-filling
        if hasattr(self.model, 'extra_regressors'):
            for regressor in self.model.extra_regressors:
                if regressor in self.training_data.columns:
                    future = future.merge(
                        self.training_data[['ds', regressor]],
                        on='ds',
                        how='left'
                    )
                    future[regressor] = future[regressor].ffill()

        self.forecast = self.model.predict(future)
        return future, self.forecast

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance on training data.

        Returns:
            Dictionary of evaluation metrics
        """
        if len(self.forecast) <= 0 or len(self.training_data) <= 0:
            raise ValueError("Model must be trained and forecast generated")

        merged = pd.merge(
            self.training_data[['ds', 'y']],
            self.forecast[['ds', 'yhat']],
            on='ds'
        )

        return {
            'mse': mean_squared_error(merged['y'], merged['yhat']),
            'mae': (merged['y'] - merged['yhat']).abs().mean(),
            'rmse': mean_squared_error(merged['y'], merged['yhat'], squared=False)
        }

    def plot_forecast(self, df, show_components: bool = True) -> None:
        """
        Visualize the forecast results.

        Args:
            show_components: Whether to show trend/seasonality components
        """
        if len(self.forecast) <= 0:
            raise ValueError("No forecast available to plot")

        plt.figure(figsize=(12, 6))
        plt.plot(df['ds'], df['y'], label='Actuals', color='black')
        plt.plot(self.forecast['ds'], self.forecast['yhat'], label='Forecast', color='blue')
        plt.fill_between(
            self.forecast['ds'], self.forecast['yhat_lower'], self.forecast['yhat_upper'],
            color='blue', alpha=0.2, label='Confidence Interval')

        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title('Actuals vs. Forecast')
        plt.legend()
        plt.grid(True)

        if show_components:
            fig2 = self.model.plot_components(self.forecast)
            plt.tight_layout()

        plt.show()


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weather Temperature Forecasting with Prophet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file with columns: ds, temperature_celsius'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save forecast results (CSV)'
    )
    parser.add_argument(
        '--periods',
        type=int,
        default=7,
        help='Number of days to forecast'
    )
    parser.add_argument(
        '--use-regressor',
        nargs='*',  # Accept 0 or more arguments
        default=None,
        help='Columns to use as additional regressors'
    )
    parser.add_argument(
        '--no-tuning',
        action='store_true',
        help='Skip hyperparameter tuning'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Number of hyperparameter optimization trials'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    forecaster = WeatherForecaster()

    # try:
    # Load and validate data
    df = forecaster.load_data(args.input)

    # Train model
    forecaster.train(
        df,
        use_regressor=args.use_regressor,
        tune_hyperparameters=not args.no_tuning
    )

    # Generate forecast
    future, forecast = forecaster.predict(periods=args.periods)

    # Show evaluation metrics
    metrics = forecaster.evaluate()
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.2f}")

    merged = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df[['ds', 'y']], on='ds', how='left')
    merged.rename(columns={'y': 'actual'}, inplace=True)
    merged['diff'] = merged['yhat'] - merged['actual']
    print(merged.tail(100))

    # Visualize results
    forecaster.plot_forecast(df)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        forecast[['ds', 'yhat']].tail(args.periods).to_csv(
            output_path,
            index=False
        )
        print(f"\nForecast saved to {output_path.resolve()}")

    # except Exception as e:
    #     print(f"\nError: {str(e)}")
    #     return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())