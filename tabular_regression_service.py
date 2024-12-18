# *coding:utf-8 -*

"""
Created on Wed. Dec. 11 2024
@author: JUN-SU PARK

Medical Data Regression Analysis Service

This module provides functionalities for:

1. Automated regression model training using AutoGluon
2. Model evaluation and performance metrics calculation
3. Visualization of regression results
4. Feature importance analysis
5. Prediction service for new data
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend for non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns

class TabularRegressionService:
    """
    A service class for handling tabular regression tasks using AutoGluon.
    
    This class manages the entire lifecycle of regression models including
    training, evaluation, visualization, and prediction.
    """

    def __init__(self):
        """
        Initialize the TabularRegressionService with necessary paths and configurations.
        """
        self.model_path = 'models/regression'  # Path for saving regression models
        self.predictor = None
        self.training_results = None
        os.makedirs(self.model_path, exist_ok=True)

    def plot_regression_results(self, actual: np.ndarray, predicted: np.ndarray, 
                              save_path: str, model_name: str = None) -> None:
        """
        Create and save visualization plots for regression results.

        Args:
            actual (np.ndarray): Array of actual target values
            predicted (np.ndarray): Array of predicted values
            save_path (str): Path where the plot will be saved
            model_name (str, optional): Name of the model for plot title

        Returns:
            None: Saves the plot to the specified path
        """
        plt.style.use('dark_background')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('#1e293b')
        
        # 1. Scatter plot of actual vs predicted values
        ax1.set_facecolor('#1e293b')
        sns.scatterplot(x=actual, y=predicted, alpha=0.6, color='#60a5fa', ax=ax1)
        
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        ax1.plot([min_val, max_val], [min_val, max_val], '--', color='#ef4444', label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values', color='white')
        ax1.set_ylabel('Predicted Values', color='white')
        # Include model name in title if provided
        title = 'Actual vs Predicted Values'
        if model_name:
            title += f' ({model_name})'
        ax1.set_title(title, color='white', pad=15)
        ax1.legend(facecolor='#1e293b', edgecolor='#4b5563')
        ax1.grid(True, linestyle='--', alpha=0.2)
        
        # 2. Residual plot
        ax2.set_facecolor('#1e293b')
        residuals = actual - predicted
        sns.scatterplot(x=predicted, y=residuals, alpha=0.6, color='#60a5fa', ax=ax2)
        ax2.axhline(y=0, color='#ef4444', linestyle='--')
        
        ax2.set_xlabel('Predicted Values', color='white')
        ax2.set_ylabel('Residuals', color='white')
        title = 'Residual Plot'
        if model_name:
            title += f' ({model_name})'
        ax2.set_title(title, color='white', pad=15)
        ax2.grid(True, linestyle='--', alpha=0.2)

        # Set border colors
        for ax in [ax1, ax2]:
            ax.spines['bottom'].set_color('#4b5563')
            ax.spines['top'].set_color('#4b5563')
            ax.spines['left'].set_color('#4b5563')
            ax.spines['right'].set_color('#4b5563')
            ax.tick_params(colors='white')  # Tick colors
        
        plt.tight_layout()
        plt.savefig(save_path, 
                    facecolor='#1e293b',  # Overall figure background color
                    edgecolor='#4b5563',  # Border color
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

    def train(self, file_path: str, label_column: str, 
              eval_metric: str = 'root_mean_squared_error',
              preset: str = 'medium_quality', 
              time_limit: int = 300,
              problem_type: str = 'regression') -> dict:
        """
        Train regression models using AutoGluon and evaluate their performance.

        Args:
            file_path (str): Path to the input data file
            label_column (str): Name of the target variable column
            eval_metric (str): Metric used for evaluation
            preset (str): AutoGluon preset configuration
            time_limit (int): Maximum training time in seconds
            problem_type (str): Type of problem (should be 'regression')

        Returns:
            dict: Dictionary containing training results, metrics, and visualizations
        """
        try:
            print(f"\n=== Starting Regression Training ===")
            print(f"Parameters:")
            print(f"- File path: {file_path}")
            print(f"- Label column: {label_column}")
            print(f"- Eval metric: {eval_metric}")
            print(f"- Preset: {preset}")
            print(f"- Time limit: {time_limit}")

            # Check if file exists before loading data
            if not os.path.exists(file_path):
                print(f"Error: File not found at {file_path}")
                return {'error': 'File not found'}

            # Load data
            try:
                data = TabularDataset(file_path)
                print(f"Data loaded successfully. Shape: {data.shape}")
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                return {'error': f'Data loading error: {str(e)}'}

            # Verify label column
            if label_column not in data.columns:
                print(f"Error: Label column '{label_column}' not found in data")
                return {'error': f'Label column "{label_column}" not found'}

            start_time = time.time()

            # Load and split data
            try:
                data = TabularDataset(file_path)
                train_data = data.sample(frac=0.8, random_state=0)
                test_data = data.drop(train_data.index)
                print(f"Data loaded and split. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            except Exception as e:
                print(f"Error in data loading: {str(e)}")
                return {'error': f'Data processing error: {str(e)}'}

            # Model training
            try:
                self.predictor = TabularPredictor(
                    label=label_column,
                    problem_type=problem_type,
                    eval_metric=eval_metric,
                    path=os.path.join(self.model_path, 'current_model')
                ).fit(
                    train_data=train_data,
                    time_limit=time_limit,
                    presets=preset
                )
                print("Model training completed")
            except Exception as e:
                print(f"Error in model training: {str(e)}")
                return {'error': f'Model training error: {str(e)}'}

            # Collect results
            predictions = self.predictor.predict(test_data)
            leaderboard = self.predictor.leaderboard(test_data, silent=True)
            feature_importance = self.predictor.feature_importance(test_data)
            performance = self.predictor.evaluate(test_data)

            # Individual model performance modification
            model_performances = {}
            best_rmse = float('inf')
            best_model_name = None
            best_predictions = None  # Storage for best model predictions
            
            for model_name in self.predictor.model_names():
                evaluation_result = self.predictor.evaluate(test_data, model=model_name)
                current_predictions = self.predictor.predict(test_data, model=model_name)
                print(f"\nEvaluation for {model_name}:")
                print(f"Raw evaluation result: {evaluation_result}")
                
                if isinstance(evaluation_result, dict):
                    rmse = abs(evaluation_result.get('root_mean_squared_error', 0.0))
                else:
                    rmse = abs(evaluation_result)
                
                rmse = float(abs(rmse))
                model_performances[model_name] = rmse
                
                # Update best RMSE and corresponding model name and predictions
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = model_name
                    best_predictions = current_predictions

            print("\nFinal Model Performances:", model_performances)
            print(f"Best RMSE: {best_rmse} (Model: {best_model_name})")

            # Regression results visualization (using predictions from best model)
            results_df = pd.DataFrame({
                'Actual': test_data[label_column],
                'Predicted': best_predictions
            })

            # Save visualization (pass best model name)
            plot_filename = 'regression_plots.png'
            plot_path = os.path.join(self.model_path, plot_filename)
            self.plot_regression_results(
                actual=results_df['Actual'],
                predicted=results_df['Predicted'],
                save_path=plot_path,
                model_name=best_model_name
            )

            # Calculate additional statistics (based on best model)
            mse = ((results_df['Actual'] - results_df['Predicted']) ** 2).mean()
            rmse = np.sqrt(mse)
            mae = abs(results_df['Actual'] - results_df['Predicted']).mean()
            r2 = 1 - (((results_df['Actual'] - results_df['Predicted']) ** 2).sum() / 
                      ((results_df['Actual'] - results_df['Actual'].mean()) ** 2).sum())

            print(f"Performance Metrics:")  # Add debug output
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"R2: {r2}")

            # Process feature importance
            feature_importance_dict = {}
            for feature in feature_importance.index:
                feature_importance_dict[str(feature)] = float(feature_importance.loc[feature, 'importance'])

            # Select top 10 features
            sorted_features = sorted(
                feature_importance_dict.items(),
                key=lambda x: abs(float(x[1])),
                reverse=True
            )[:10]
            feature_importance_dict = {k: v for k, v in sorted_features}

            # Save results with best_model_name added
            self.training_results = {
                'leaderboard': leaderboard.to_dict(orient='records'),
                'feature_importance': feature_importance_dict,
                'sample_predictions': results_df.head(10).to_dict(orient='records'),
                'performance_metrics': {
                    'rmse': float(best_rmse),
                    'mse': float(abs(mse)),
                    'mae': float(abs(mae)),
                    'r2': float(r2),
                    'best_model': best_model_name  # Add best model name
                },
                'model_performances': model_performances,
                'training_time': float(time.time() - start_time),
                'plot_path': f'/models/regression/{plot_filename}'
            }

            return self.training_results

        except Exception as e:
            import traceback
            print(f"Unexpected error in training:")
            print(traceback.format_exc())
            return {'error': str(e)}

    def get_results(self) -> dict:
        """
        Retrieve the current training results.

        Returns:
            dict: Dictionary containing the latest training results or error message
        """
        if self.training_results is None:
            return {'error': 'No training results available'}
        
        print("\n=== Regression Service Debug ===")
        print("Training results type:", type(self.training_results))
        print("Training results keys:", self.training_results.keys() if isinstance(self.training_results, dict) else "Not a dict")
        print("Full training results:", self.training_results)
        print("==============================\n")
        
        return self.training_results

    def predict(self, data: pd.DataFrame) -> dict:
        """
        Make predictions on new data using the trained model.

        Args:
            data (pd.DataFrame): Input data for prediction

        Returns:
            dict: Dictionary containing predictions or error message
        """
        if self.predictor is None:
            return {'error': 'Model not trained'}
        try:
            predictions = self.predictor.predict(data)
            return {
                'predictions': predictions.tolist()
            }
        except Exception as e:
            return {'error': str(e)}