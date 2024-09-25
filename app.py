#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:40:18 2024

@author: preethamu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop Price Predictor
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

app = Flask(__name__)

# Load and prepare the dataset
file_path = 'crop_price.csv'  # Ensure this file is in the same directory
crop_df = pd.read_csv(file_path)

# Convert 'arrival_date' to datetime, handling multiple date formats
crop_df['arrival_date'] = pd.to_datetime(crop_df['arrival_date'], dayfirst=True, errors='coerce')

# Drop rows with invalid dates
crop_df = crop_df.dropna(subset=['arrival_date'])

# Extract date features
crop_df['day'] = crop_df['arrival_date'].dt.day
crop_df['month'] = crop_df['arrival_date'].dt.month
crop_df['year'] = crop_df['arrival_date'].dt.year

# Drop unnecessary columns
crop_df = crop_df.drop(columns=['arrival_date', 'variety', 'district', 'min_price', 'max_price'])

# Encode categorical features
le_state = LabelEncoder()
le_commodity = LabelEncoder()

crop_df['state'] = le_state.fit_transform(crop_df['state'])
crop_df['commodity'] = le_commodity.fit_transform(crop_df['commodity'])

# Separate features and target variable
X = crop_df[['state', 'commodity', 'day', 'month', 'year']]
y = crop_df['avg_price']

# Apply log transformation to the target variable
y_log = np.log1p(y)

# Train the XGBoost model
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
model.fit(X, y_log)

# Define the Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the commodity name from the form
        commodity_name = request.form.get('commodity')

        if not commodity_name:
            return jsonify({'error': 'Please enter a commodity name.'}), 400

        # Check if the commodity exists in the LabelEncoder
        try:
            commodity_encoded = le_commodity.transform([commodity_name])[0]
        except ValueError:
            return jsonify({'error': f'Commodity "{commodity_name}" not found in the dataset.'}), 400

        # Get the future date (3 months from now)
        future_date = datetime.now() + timedelta(days=90)
        day = future_date.day
        month = future_date.month
        year = future_date.year

        # Fixed state as 'Karnataka'
        state_encoded = le_state.transform(['Karnataka'])[0]

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'state': [state_encoded],
            'commodity': [commodity_encoded],
            'day': [day],
            'month': [month],
            'year': [year]
        })

        # Make the prediction
        predicted_log_price = model.predict(input_data)
        predicted_price = np.expm1(predicted_log_price)[0]

        # Round the predicted price to two decimal places and convert to float
        predicted_price = float(round(predicted_price, 2))

        # Format the future date
        future_date_str = future_date.strftime('%Y-%m-%d')

        # Return the result
        return jsonify({
            'commodity': commodity_name,
            'predicted_price': predicted_price,  # Now it's a float
            'date': future_date_str,
            'location': 'Karnataka'
        })
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
