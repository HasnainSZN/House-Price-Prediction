from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = joblib.load('house_price_model.joblib')
scaler = joblib.load('house_price_scaler.joblib')

@app.route('/predict_price', methods=['POST'])
def predict_house_price():
    try:
        # Expected input: square_feet, bedrooms, bathrooms, age_years, distance_city_center
        data = request.json
        features = [
            data['square_feet'], 
            data['bedrooms'], 
            data['bathrooms'], 
            data['age_years'], 
            data['distance_city_center']
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict price
        predicted_price = model.predict(features_scaled)[0]
        
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failure'
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)