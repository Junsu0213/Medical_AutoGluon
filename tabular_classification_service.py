# *coding:utf-8 -*

"""
Created on Mon. Dec. 16 2024
@author: JUN-SU PARK

Medical Data Classification Analysis Service

This module provides functionalities for:

1. Automated classification model training using AutoGluon
2. Model evaluation and performance metrics calculation
3. Confusion matrix analysis
4. ROC curve analysis
5. Feature importance analysis
6. Prediction service for new data
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import time
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend for non-GUI environments

class TabularClassificationService:
    """
    A service class for handling tabular classification tasks using AutoGluon.
    
    This class manages the entire lifecycle of classification models including
    training, evaluation, metrics calculation, and prediction.
    """

    def __init__(self):
        """
        Initialize the TabularClassificationService with necessary paths and configurations.
        """
        self.model_path = 'models/classification'  # Path for saving classification models
        self.predictor = None
        self.training_results = None
        os.makedirs(self.model_path, exist_ok=True)

    def train(self, file_path: str, label_column: str, 
              eval_metric: str = 'accuracy',
              preset: str = 'medium_quality', 
              time_limit: int = 300) -> dict:
        """
        Train classification models using AutoGluon and evaluate their performance.

        Args:
            file_path (str): Path to the input data file
            label_column (str): Name of the target variable column
            eval_metric (str): Metric used for evaluation
            preset (str): AutoGluon preset configuration
            time_limit (int): Maximum training time in seconds

        Returns:
            dict: Dictionary containing training results, metrics, and visualizations
        """
        try:
            print(f"Training started with parameters:")
            print(f"- File: {file_path}")
            print(f"- Label: {label_column}")
            print(f"- Metric: {eval_metric}")
            print(f"- Preset: {preset}")
            print(f"- Time limit: {time_limit}")

            # Record training start time
            start_time = time.time()

            # Load data
            try:
                data = TabularDataset(file_path)
                print(f"Data loaded successfully. Shape: {data.shape}")
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                return {'error': f'Data loading error: {str(e)}'}

            # Split data
            try:
                train_data = data.sample(frac=0.8, random_state=0)
                test_data = data.drop(train_data.index)
                print(f"Data split completed. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            except Exception as e:
                print(f"Error splitting data: {str(e)}")
                return {'error': f'Data splitting error: {str(e)}'}

            # Train model
            try:
                self.predictor = TabularPredictor(
                    label=label_column,
                    eval_metric=eval_metric,
                    path=os.path.join(self.model_path, 'current_model')
                ).fit(
                    train_data=train_data,
                    time_limit=time_limit,
                    presets=preset
                )
                print("Model training completed successfully")
            except Exception as e:
                print(f"Error during model training: {str(e)}")
                return {'error': f'Model training error: {str(e)}'}

            # Collect results
            leaderboard = self.predictor.leaderboard(test_data, silent=True)
            feature_importance = self.predictor.feature_importance(test_data)
            predictions = self.predictor.predict(test_data)
            pred_probs = self.predictor.predict_proba(test_data)

            # Check pred_probs data structure
            print("Pred probs type:", type(pred_probs))
            print("Pred probs first element type:", type(pred_probs[0]) if len(pred_probs) > 0 else "empty")
            print("Pred probs first few values:", pred_probs[:3])

            performance = self.predictor.evaluate(test_data, auxiliary_metrics=True)

            # Check feature importance data structure
            print("Feature importance type:", type(feature_importance))
            print("Feature importance shape:", feature_importance.shape if hasattr(feature_importance, 'shape') else 'No shape')
            print("Feature importance raw:", feature_importance)

            # Calculate Confusion Matrix
            try:
                # Convert actual and predicted values to integers
                y_true = test_data[label_column].astype(int)
                y_pred = predictions.astype(int)
                
                # Check all possible class labels
                all_classes = sorted(list(set(y_true) | set(y_pred)))
                print("Unique classes:", all_classes)
                n_classes = len(all_classes)
                print(f"Number of classes: {n_classes}")
                
                # Calculate confusion matrix (for all classes)
                conf_matrix = confusion_matrix(
                    y_true, 
                    y_pred,
                    labels=all_classes  # Use all class labels
                ).tolist()
                
                print("Confusion Matrix Shape:", len(conf_matrix), "x", len(conf_matrix[0]))
                print("Confusion Matrix:", conf_matrix)
                print("Class Labels:", all_classes)
                
            except Exception as e:
                print(f"Error calculating confusion matrix: {str(e)}")
                # Default value on error (2x2 matrix)
                conf_matrix = [[0] * 2 for _ in range(2)]

            # Process leaderboard data
            leaderboard_dict = []
            for idx, row in leaderboard.iterrows():
                leaderboard_dict.append({
                    'model': str(row['model']),
                    'score_val': float(row['score_val']),
                    'fit_time': float(row['fit_time']),
                    'pred_time': float(row['pred_time_val'])
                })

            # Process feature importance data
            try:
                # Convert feature importance to dictionary
                feature_importance_dict = {}
                # Get importance values from DataFrame
                for feature in feature_importance.index:
                    feature_importance_dict[str(feature)] = float(feature_importance.loc[feature, 'importance'])

                # Convert numpy int64 to Python int
                for key in feature_importance_dict:
                    if hasattr(feature_importance_dict[key], 'item'):
                        feature_importance_dict[key] = feature_importance_dict[key].item()
                    else:
                        feature_importance_dict[key] = float(feature_importance_dict[key])
                
                # Select top 10 features (sorted by absolute value)
                sorted_features = sorted(
                    feature_importance_dict.items(),
                    key=lambda x: abs(float(x[1])),  # Sort by absolute value while preserving original values
                    reverse=True
                )[:10]
                # Create dictionary while preserving original values
                feature_importance_dict = {k: v for k, v in sorted_features}

                print("Feature importance dict:", feature_importance_dict)
            except Exception as e:
                print(f"Error processing feature importance: {str(e)}")
                print(f"Error traceback: {traceback.format_exc()}")
                feature_importance_dict = {}

            # Check data types before collecting results
            print("Data types:")
            print(f"Label column type: {test_data[label_column].dtype}")
            print(f"Predictions type: {type(predictions)}")
            print(f"Probabilities type: {type(pred_probs)}")
            print("Sample values:")
            print(f"Label: {test_data[label_column].head(10)}")  # Increased to 10 samples
            print(f"Predictions: {predictions[:10]}")            # Increased to 10 samples
            print(f"Probabilities: {pred_probs[:10]}")          # Increased to 10 samples
            
            # Update individual model performance section
            model_performances = {}
            for model_name in self.predictor.model_names():  # Add parentheses
                model_performance = self.predictor.evaluate(test_data, model=model_name)
                model_performances[model_name] = model_performance

            # Calculate ROC curve data
            try:
                # Calculate ROC curve only for binary classification
                unique_labels = len(set(test_data[label_column]))
                if unique_labels == 2:
                    y_true = test_data[label_column].values
                    y_score = pred_probs.iloc[:, 1].values  # probability of positive class
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    roc_auc = auc(fpr, tpr)
                    
                    roc_data = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'auc': float(roc_auc)
                    }
                else:
                    roc_data = None
            except Exception as e:
                print(f"Error calculating ROC curve: {str(e)}")
                roc_data = None

            # Structure results
            self.training_results = {
                'leaderboard': leaderboard_dict,
                'feature_importance': feature_importance_dict,
                'sample_predictions': [{
                    'Actual': int(float(actual)),
                    'Predicted': int(float(pred)),
                    'Probability': float(max(prob) if isinstance(prob, np.ndarray) else prob)
                } for actual, pred, prob in zip(
                    test_data[label_column].values[:10],
                    predictions.values[:10],
                    [row for row in pred_probs[:10].values]
                )],
                'performance_metrics': {
                    'accuracy': float(performance['accuracy']),
                    'f1': float(performance.get('f1', 0.0)),
                    'roc_auc': float(performance.get('roc_auc', 0.0))
                },
                'confusion_matrix': conf_matrix,
                'training_time': float(time.time() - start_time),
                'model_performances': model_performances,
                'roc_data': roc_data  # Add ROC curve data
            }
            print("Training results saved:", self.training_results)  # Verify results are saved
            return self.training_results

        except Exception as e:
            print(f"Error during result processing: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {'error': f'Result processing error: {str(e)}'}

    def get_results(self):
        """
        Retrieve the current training results.

        Returns:
            dict: Dictionary containing training results or error message
        """
        return self.training_results if self.training_results else {'error': 'No training results available'}

    def predict(self, data):
        """
        Make predictions on new data using the trained model.

        Args:
            data: Input data for prediction

        Returns:
            dict: Dictionary containing predictions and probabilities or error message
        """
        if self.predictor is None:
            return {'error': 'Model not trained'}
        try:
            predictions = self.predictor.predict(data)
            probabilities = self.predictor.predict_proba(data)
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist()
            }
        except Exception as e:
            return {'error': str(e)} 