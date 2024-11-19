import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_house_data():
    # Synthetic house price dataset
    np.random.seed(42)
    data = {
        'square_feet': np.random.randint(800, 5000, 500),
        'bedrooms': np.random.randint(1, 6, 500),
        'bathrooms': np.random.randint(1, 4, 500),
        'age_years': np.random.randint(0, 50, 500),
        'distance_city_center': np.random.randint(1, 20, 500),
        'price': np.zeros(500)
    }
    
    # Create a simplistic price calculation model
    df = pd.DataFrame(data)
    df['price'] = (
        df['square_feet'] * 200 +  # Value per square foot
        df['bedrooms'] * 50000 +   # Bedroom premium
        df['bathrooms'] * 30000 -  # Bathroom value 
        df['age_years'] * 1000 -   # Age depreciation
        df['distance_city_center'] * 2000  # Distance discount
    ) + np.random.normal(0, 50000, 500)  # Add some noise
    
    return df

def train_house_price_model():
    # Load dataset
    df = load_house_data()
    
    # Separate features and target
    X = df[['square_feet', 'bedrooms', 'bathrooms', 'age_years', 'distance_city_center']]
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate and print performance
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Train R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    # Save model and scaler
    joblib.dump(model, 'house_price_model.joblib')
    joblib.dump(scaler, 'house_price_scaler.joblib')

if __name__ == '__main__':
    train_house_price_model()