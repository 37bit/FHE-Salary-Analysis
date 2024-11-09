import json
import os
import numpy as np
from typing import Dict, Tuple, List

def save_model(role: str, coefficients: np.ndarray, feature_names: List[str], scaler_info: Dict, salary_scaler: Dict):
    """Save model coefficients, feature names, and scaling information"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_data = {
        'coefficients': coefficients.tolist(),
        'feature_names': feature_names,
        'scaler_info': {
            'features': scaler_info['features'],
            'bounds': scaler_info['bounds']
        },
        'salary_scaler': salary_scaler  # Add salary scaling info
    }
    
    with open(f'models/{role.lower()}_model.json', 'w') as f:
        json.dump(model_data, f, indent=4)
    print(f"Model saved for role: {role}")

def predict_salary(role: str, input_data: Dict[str, float]) -> float:
    try:
        model_path = f'models/{role.lower()}_model.json'
        if not os.path.exists(model_path):
            print(f"No model file found for role: {role}")
            return None

        with open(model_path, 'r') as f:
            model_data = json.load(f)

        coefficients = np.array(model_data['coefficients'])
        feature_names = model_data['feature_names']
        scaler_info = model_data['scaler_info']
        salary_scaler = model_data['salary_scaler']

        # Add derived features with safe computation
        input_data['experience_ratio'] = input_data.get('Years_of_Experience', 0) / max((input_data.get('Age', 0) - 18), 1)
        input_data['experience_ratio'] = min(max(input_data['experience_ratio'], 0), 1)

        # Scale numeric features
        numeric_features = scaler_info['features']
        for feature, (min_val, max_val) in zip(numeric_features, scaler_info['bounds']):
            if feature in input_data:
                range_val = max_val - min_val if max_val - min_val != 0 else 1
                # Apply scaling to [0, 1] and clip
                input_data[feature] = (input_data[feature] - min_val) / range_val
                input_data[feature] = min(max(input_data[feature], 0), 1)
            else:
                input_data[feature] = 0.5  # Default value if missing

        # Ensure binary features are set
        for feature in feature_names:
            if feature not in input_data:
                input_data[feature] = 0

        # Create feature vector
        X = np.array([input_data.get(feature, 0) for feature in feature_names])

        # Make prediction
        prediction = np.dot(X, coefficients)

        # Rescale prediction back to original salary scale
        min_salary = salary_scaler['min']
        max_salary = salary_scaler['max']
        salary_range = max_salary - min_salary if max_salary - min_salary != 0 else 1
        prediction = prediction * salary_range + min_salary

        return prediction

    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None