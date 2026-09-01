from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)

# Global variables to store models and scaler
models = {}
scaler = None
feature_columns = None
categorical_columns = None

def train_models():
    """Train all models and store them"""
    global models, scaler, feature_columns, categorical_columns
    
    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), 'fifa_world_cup_2026_player_performance (1).csv')
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Preprocessing
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100)**2)
    df.drop_duplicates(inplace=True)
    
    # Convert categorical columns
    for col in ['nationality', 'team', 'position', 'preferred_foot', 'club_name', 'stadium', 'city', 'opponent_team', 'tournament_stage', 'match_result']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Extract date features
    df['match_year'] = df['match_date'].dt.year
    df['match_month'] = df['match_date'].dt.month
    df['match_day_of_week'] = df['match_date'].dt.dayofweek
    df['match_day_of_month'] = df['match_date'].dt.day
    df['match_week_of_year'] = df['match_date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['match_day_of_week'].isin([5, 6]).astype(int)
    
    # Drop rows with missing match_date
    df.dropna(subset=['match_date'], inplace=True)
    
    # Remove outliers using IQR
    numerical_cols_to_check = df.select_dtypes(include=['number']).columns
    for col in numerical_cols_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Prepare features and target
    y = df['performance_score']
    features_to_exclude = ['player_id', 'player_name', 'match_id', 'match_date', 'performance_score']
    X = df.drop(columns=features_to_exclude, errors='ignore')
    
    # Store categorical columns info
    categorical_columns = X.select_dtypes(include=['category']).columns.tolist()
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    feature_columns = X.columns.tolist()
    
    # Scale numerical columns
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['number']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Drop rows with NaN
    df_combined = X.copy()
    df_combined['performance_score'] = y
    df_combined.dropna(inplace=True)
    
    y = df_combined['performance_score']
    X = df_combined.drop('performance_score', axis=1)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    model_names = {
        'Linear Regression': Ridge(alpha=0.01, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'KNN': KNeighborsRegressor(),
        'SVM': SVR(),
        'Ridge Regression': Ridge(random_state=42)
    }
    
    for name, model in model_names.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        models[name] = {
            'model': model,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    print("Models trained successfully!")

def preprocess_input(input_data):
    """Preprocess user input to match training data format"""
    global feature_columns, scaler
    
    # Create a DataFrame with the input
    df_input = pd.DataFrame([input_data])
    
    # One-hot encode categorical variables
    categorical_cols = df_input.select_dtypes(include=['object']).columns.tolist()
    df_input = pd.get_dummies(df_input, drop_first=True)
    
    # Ensure all columns from training are present
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Reorder columns to match training data
    df_input = df_input[feature_columns]
    
    # Scale numerical columns
    numerical_cols = df_input.select_dtypes(include=['number']).columns
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
    
    return df_input

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions"""
    try:
        data = request.json
        
        # Preprocess input
        X_input = preprocess_input(data)
        
        # Make predictions with all models
        predictions = {}
        for name, model_info in models.items():
            model = model_info['model']
            pred = model.predict(X_input)[0]
            predictions[name] = round(float(pred), 2)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'best_model': max(predictions.items(), key=lambda x: models[x[0]]['r2'])[0]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """API endpoint to get model information"""
    info = {}
    for name, model_data in models.items():
        info[name] = {
            'r2': round(model_data['r2'], 4),
            'rmse': round(model_data['rmse'], 4),
            'mae': round(model_data['mae'], 4)
        }
    return jsonify(info)

if __name__ == '__main__':
    # Train models on startup
    train_models()
    app.run(debug=True, port=5000)
