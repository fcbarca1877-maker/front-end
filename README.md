# FIFA 2026 Player Performance Predictor - Frontend

A web-based application to predict FIFA World Cup 2026 player performance scores using machine learning models.

## Features

- **Interactive Web Interface**: User-friendly form to input player statistics
- **Multiple ML Models**: Predictions from 5 different models (Linear Regression, Decision Tree, KNN, SVM, Ridge Regression)
- **Best Model Selection**: Automatically shows the best prediction based on model performance
- **Model Metrics**: View R² score, RMSE, and MAE for each model
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
front end/
├── app.py                              # Flask backend server
├── requirements.txt                    # Python dependencies
├── templates/
│   └── index.html                      # Web interface
└── fifa_world_cup_2026_player_performance (1).csv  # Training data
```

## Installation & Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 3. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

## How to Use

1. **Enter Player Information**:
   - Age (16-45)
   - Height (cm) and Weight (kg)
   - Playing Position
   - Goals, Assists, Shots, Clean Sheets
   - Expected Goals (xG) and Tournament Rating
   - Preferred Foot

2. **Click "Predict Performance"**: The system will analyze the data and provide predictions

3. **View Results**:
   - Best predicted performance score
   - Predictions from all models
   - Detailed model performance metrics (R², RMSE, MAE)

## Technical Details

### Backend (Flask)
- **app.py**: Handles model training, data preprocessing, and prediction API
- Trains 5 regression models on startup
- Preprocesses input data to match training format
- Provides REST API endpoints:
  - `GET /` - Serves the HTML interface
  - `POST /api/predict` - Makes predictions
  - `GET /api/model-info` - Returns model performance metrics

### Frontend (HTML/CSS/JavaScript)
- **index.html**: Interactive web interface with real-time validation
- Responsive grid layout
- Smooth animations and loading states
- Color-coded results display

### Data Processing
1. Load FIFA World Cup 2026 player performance data
2. Handle missing values and outliers (IQR method)
3. Feature engineering (BMI calculation, date extraction)
4. Categorical encoding (one-hot encoding)
5. Feature scaling (StandardScaler)
6. Train-test split (80-20)

### Models
1. **Linear Regression (Ridge α=0.01)** - Best baseline model
2. **Decision Tree** - Captures non-linear relationships
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Support Vector Machine (SVM)** - Non-linear regression
5. **Ridge Regression** - Regularized linear model

## Performance Metrics Explained

- **R² Score**: Measures how well predictions fit actual values (0-1, higher is better)
- **RMSE**: Root Mean Squared Error - average prediction error in performance score
- **MAE**: Mean Absolute Error - average absolute prediction error

## Features Used for Prediction

- Player demographics: Age, Height, Weight, BMI
- Performance statistics: Goals, Assists, Shots, Clean Sheets
- Quality metrics: Expected Goals (xG), Tournament Rating
- Player attributes: Position, Preferred Foot

## Example Usage

1. Input a player with:
   - Age: 28
   - Height: 182 cm
   - Weight: 78 kg
   - Position: Forward
   - Goals: 8
   - Assists: 3
   - Shots: 25
   - Clean Sheets: 0
   - Expected Goals: 4.2
   - Tournament Rating: 7.8

2. Click "Predict Performance"

3. Get instant predictions with model comparison

## Troubleshooting

**Issue**: Port 5000 already in use
- Change the port in `app.py` line `app.run(debug=True, port=5001)`

**Issue**: CSV file not found
- Ensure `fifa_world_cup_2026_player_performance (1).csv` is in the same directory as `app.py`

**Issue**: Missing dependencies
- Run: `pip install -r requirements.txt` again

**Issue**: Models not training
- Check that the CSV file has all required columns
- Verify data format (numerical columns should be numeric)

## Requirements

- Python 3.8+
- Flask 2.3.2+
- pandas 2.0.3+
- numpy 1.24.3+
- scikit-learn 1.3.0+

## License

This project is for educational purposes.

## Notes

- The application trains all models on startup (may take 30-60 seconds on first run)
- Predictions are based on the trained models and historical data patterns
- Model performance metrics are displayed to help assess prediction reliability
- Input validation ensures realistic player statistics
