# House Price Prediction AI Model

## Project Overview
A full-stack machine learning application for predicting house prices using Random Forest Regression, with a Flask backend API and React frontend.

## Features
- Machine learning model for house price estimation
- Flask-based REST API for model predictions
- Interactive React web interface
- Feature-based price prediction

## Technology Stack
- Backend: Python, Flask, Scikit-learn
- Frontend: React, Axios
- ML Model: Random Forest Regressor

## Prerequisites
- Python 3.8+
- Node.js 14+
- pip
- npm



## Model Features
The price prediction considers:
- Square footage
- Number of bedrooms
- Number of bathrooms
- House age
- Distance from city center

## Project Structure
```
house-price-predictor/
│
├── backend/
│   ├── model.py
│   ├── main.py
│   ├── house_price_model.joblib
│   └── house_price_scaler.joblib
│
├── frontend/
│   ├── src/
│   │   └── index.jsx
│   └── package.json
│
└── README.md
```

## Model Performance
- Utilizes Random Forest Regressor
- Includes feature scaling
- Synthetic dataset for demonstration

## Deployment Notes
- Use environment variables for configuration
- Implement proper error handling
- Consider adding authentication

## Future Improvements
- Real-world dataset integration
- More advanced feature engineering
- Enhanced error handling
- Containerization with Docker

