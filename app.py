# *coding:utf-8 -*

"""
Created on Mon. Dec. 9 2024
@author: JUN-SU PARK

Medical Data Analysis Web Application

This module provides the web interface and API endpoints for:

1. File upload and data processing
2. Model training configuration
3. Results visualization
4. API endpoints for model interaction
5. Serving static files and model results
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from tabular_classification_service import TabularClassificationService
from tabular_regression_service import TabularRegressionService
import pandas as pd
import os
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Create AutoML service instances
classification_service = TabularClassificationService()
regression_service = TabularRegressionService()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tabular/classification', methods=['GET', 'POST'])
def tabular_classification():
    if request.method == 'POST':
        print("=== Classification POST request received ===")
        print("Form data:", request.form)
        print("Files:", request.files)
        
        try:
            # Handle file upload
            file = request.files.get('data_file')
            if not file:
                return jsonify({'error': 'No file uploaded'}), 400

            # Save file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Validate CSV file
            try:
                data = pd.read_csv(file_path)
            except Exception as e:
                return jsonify({'error': f'CSV file reading error: {str(e)}'}), 400

            # Get form data
            label_column = request.form.get('label_column')
            if not label_column:
                return jsonify({'error': 'Label column is required'}), 400
            
            # Check if label column exists in the CSV file
            if label_column not in data.columns:
                return jsonify({'error': f'Label column "{label_column}" not found in CSV file'}), 400

            eval_metric = request.form.get('eval_metric', 'accuracy')
            preset = request.form.get('preset', 'medium_quality')
            try:
                time_limit = int(request.form.get('time_limit', 300))
            except ValueError:
                return jsonify({'error': 'Invalid time limit value'}), 400

            # Start model training
            results = classification_service.train(
                file_path=file_path,
                label_column=label_column,
                eval_metric=eval_metric,
                preset=preset,
                time_limit=time_limit
            )

            if 'error' in results:
                return jsonify({'error': results['error']}), 500

            # Redirect to results page after training is complete
            return jsonify({'success': True, 'redirect': '/classification-results'})

        except Exception as e:
            import traceback
            print("Error:", str(e))
            print("Traceback:", traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    # GET request
    return render_template('tabular_classification.html')

@app.route('/tabular/regression', methods=['GET', 'POST'])
def tabular_regression():
    if request.method == 'POST':
        try:
            # Handle file upload
            file = request.files.get('data_file')
            if not file:
                print("Error: No file uploaded")
                return jsonify({'error': 'No file uploaded'}), 400

            # Save file path
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"Saving file to: {file_path}")
            file.save(file_path)

            # Get form data
            label_column = request.form.get('label_column')
            eval_metric = request.form.get('eval_metric')
            preset = request.form.get('preset')
            time_limit = int(request.form.get('time_limit', 300))
            
            print(f"Form data received:")
            print(f"- Label column: {label_column}")
            print(f"- Eval metric: {eval_metric}")
            print(f"- Preset: {preset}")
            print(f"- Time limit: {time_limit}")

            # Start model training
            results = regression_service.train(
                file_path=file_path,
                label_column=label_column,
                eval_metric=eval_metric,
                preset=preset,
                time_limit=time_limit,
                problem_type='regression'
            )

            if 'error' in results:
                print(f"Training error: {results['error']}")
                return jsonify({'error': results['error']}), 500

            return jsonify({'success': True, 'redirect': '/tabular/regression-results'})

        except Exception as e:
            import traceback
            print("Error occurred:")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    return render_template('tabular_regression.html')

@app.route('/image/classification')
def image_classification():
    return "Image Classification 기능 수행 중입니다."

@app.route('/image/object-detection')
def image_object_detection():
    return "Image Object Detection 기능 수행 중입니다."

@app.route('/tabular/classification-results')
def tabular_classification_results():
    return render_template('tabular_classification_result.html')

@app.route('/tabular/regression-results')
def tabular_regression_results():
    return render_template('tabular_regression_result.html')

@app.route('/api/tabular/classification-results')
def get_classification_results():
    """학습 결과 조회 API"""
    results = classification_service.get_results()
    print("=== Classification Results ===")
    print("Results type:", type(results))
    print("Results content:", results)
    return jsonify(results)

@app.route('/api/tabular/regression-results')
def get_regression_results():
    """회귀 분석 결과 조회 API"""
    results = regression_service.get_results()
    print("\n=== Regression Results Debug ===")
    print("Results type:", type(results))
    print("Results keys:", results.keys() if isinstance(results, dict) else "Not a dict")
    print("Sample predictions:", results.get('sample_predictions') if isinstance(results, dict) else None)
    print("Performance metrics:", results.get('performance_metrics') if isinstance(results, dict) else None)
    print("Full results:", results)
    print("================================\n")
    
    if isinstance(results, dict) and 'error' in results:
        return jsonify({'error': results['error']}), 400
        
    return jsonify(results)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        problem_type = data.get('problem_type', 'classification')
        df = pd.DataFrame(data['data'])
        
        if problem_type == 'classification':
            predictions = classification_service.predict(df)
        else:
            predictions = regression_service.predict(df)
            
        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/<path:filename>')
def serve_model_file(filename):
    return send_from_directory(app.config['MODEL_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
