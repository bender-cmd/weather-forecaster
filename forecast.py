#!/usr/bin/env python3
"""
Weather Temperature Forecasting with Prophet

Usage: forecast.py --input <file.csv> [--output <forecast.csv>] [--periods <days>] [--use-regressor [<regressor>]] [
--plot-components] [--no-tuning] [--trials]
        forecast.py (-h | --help)

Options:
  --input <file.csv>                Path to input CSV (required).
  --output <forecast.csv>           Save forecast to this file [default: None].
  --periods <days>                  Days to forecast [default: 7].
  --use-regressor <regressor>       Regressors to use [default: humidity, pressure].
  -h --help                         Show this screen.
  --plot-components                 Whether to show component plots.
  --no-tuning                       Whether to skip hyperparameter tuning
  --trials                          Number of hyperparameter optimization trials
"""

import argparse
import warnings
from typing import Optional, Dict, Tuple, List
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import optuna
from prophet.diagnostics import cross_validation, performance_metrics


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
            # Use low_memory=False for faster CSV parsing
            df = pd.read_csv(filepath, low_memory=False)

            # Convert to datetime with infer_datetime_format for speed
            df['ds'] = pd.to_datetime(df['ds'], infer_datetime_format=True)

            # Basic data validation
            if 'temperature_celsius' not in df.columns:
                raise ValueError("Input data must contain 'temperature_celsius' column")

            # Downcast numeric columns
            df['temperature_celsius'] = pd.to_numeric(df['temperature_celsius'],
                                                      downcast='float')

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

    def _early_stopping_callback(self, study, trial):  # exit if results are not improving
        if trial.number > 100 and study.best_value > 2.0:  # Adjust threshold
            raise optuna.exceptions.TrialPruned()

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
                    'seasonality_mode', ['additive', 'multiplicative']),  # testing both modes to determine which
                # fits the temperature data better
                'holidays_prior_scale': trial.suggest_float(
                    'holidays_prior_scale', 0.1, 10.0)  # controls how strongly holidays affect temperatures
            }

            model = self._instantiate_model(params)
            model.fit(df)

            # Calculate safe cross-validation parameters
            min_date = df['ds'].min()
            max_date = df['ds'].max()
            total_days = (max_date - min_date).days

            # Perform Cross-validation, Dynamic configuration based on data size
            if total_days <= 366:  # ~1 year of data
                initial = '335 days'  # Train on first 300 days (~10 months)
                period = '30 days'  # Create monthly validation folds
                horizon = '7 days'  # Validate on last 30 days
            else:
                initial = '365 days'  # Full year base
                period = '90 days'
                horizon = '30 days'

            df_cv = cross_validation(
                model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes"
            )

            return performance_metrics(df_cv)['rmse'].mean()

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(
                                        n_startup_trials=10,  # First 10 trials random
                                        multivariate=True  # Smarter parameter combinations
                                    )
                                    )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[self._early_stopping_callback],
                       n_jobs=4  # Parallel trials (if you have multiple cores)
                       )
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
        best_params = self._tune_hyperparameters(df, n_trials) if tune_hyperparameters else None

        # Initialize model with best parameters
        self.model = self._instantiate_model(best_params)

        # Add regressors if requested and valid
        if isinstance(use_regressor, list):  # if regressors requested
            # Pre-filter valid regressors before adding
            valid_regressors = [
                col for col in (use_regressor or ['humidity', 'pressure'])
                if self._should_use_regressor(df, col)
            ]
            for col in valid_regressors[:3]:  # Limit to top 3 regressors
                self.model.add_regressor(col)
            # if len(use_regressor) > 0:  # if regressors defined
            #     for regressor_col in use_regressor:
            #         if self._should_use_regressor(df, regressor_col):
            #             self.model.add_regressor(regressor_col)
            # else:  # try defaults
            #     if self._should_use_regressor(df, 'humidity'):
            #         self.model.add_regressor('humidity')
            #     if self._should_use_regressor(df, 'pressure'):
            #         self.model.add_regressor('pressure')

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

    def plot_forecast(self, df, plot_components: bool = False, last_n_days: int = 7):
        """
        Visualize the forecast results with option to plot components.

        Args:
            df: Historical data DataFrame
            plot_components: Whether to show trend/seasonality components
            last_n_days: Number of most recent forecast days to plot (default: 7)
        """
        if len(self.forecast) <= 0:
            raise ValueError("No forecast available to plot")

        # Create main plot (full range)
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
        plt.show()

        # Create second plot (recent days only)
        if last_n_days and len(self.forecast) > last_n_days:
            plt.figure(figsize=(12, 6))

            # Define start date for plotting
            start_date = self.forecast['ds'].iloc[-1] - pd.Timedelta(days=2 * last_n_days)

            # Historical actuals starting from start_date
            last_history = df[df['ds'] >= start_date]

            # Forecast starting from start_date
            extended_forecast = self.forecast[self.forecast['ds'] >= start_date]

            # Plot actuals
            plt.plot(last_history['ds'], last_history['y'],
                     label='Actuals', color='black', marker='o')

            # Plot forecast
            plt.plot(extended_forecast['ds'], extended_forecast['yhat'],
                     label='Forecast', color='blue', marker='o')

            plt.fill_between(
                extended_forecast['ds'],
                extended_forecast['yhat_lower'],
                extended_forecast['yhat_upper'],
                color='blue', alpha=0.2, label='Confidence Interval'
            )

            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.title(f'Forecast Covering Last {2 * last_n_days} Days')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        if plot_components:
            self.model.plot_components(self.forecast)
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
    parser.add_argument(
        '--plot-components',
        action='store_true',
        help='Whether to show component plots'
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
        tune_hyperparameters=not args.no_tuning,
        n_trials=args.trials
    )

    # Generate forecast
    future, forecast = forecaster.predict(periods=args.periods)

    # Show evaluation metrics
    # metrics = forecaster.evaluate()
    # print("\nModel Performance Metrics:")
    # for metric, value in metrics.items():
    #     print(f"{metric.upper()}: {value:.2f}")

    merged = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df[['ds', 'y']], on='ds', how='left')
    merged.rename(columns={'y': 'actual'}, inplace=True)
    merged['diff'] = merged['yhat'] - merged['actual']
    print(merged.tail(100))

    # Visualize results
    forecaster.plot_forecast(df, args.plot_components, last_n_days=args.periods)

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
