import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3
import os
from datetime import datetime

app = Flask(__name__, static_folder='.')
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model and scaler
try:
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print(f"Model and scaler loaded successfully at {datetime.now()}")
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e} at {datetime.now()}")
    rf_model, scaler = None, None

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS health_data
                 (id INTEGER, timestamp TEXT, steps REAL, distance REAL, calories REAL, weight REAL, bmi REAL,
                  heart_rate REAL, mobile_usage REAL, water_intake REAL, systolic_bp REAL, diastolic_bp REAL)''')
    conn.commit()
    conn.close()

init_db()

# Process health data and calculate stress
def process_health_data(data):
    if not rf_model or not scaler:
        return {"error": "Model or scaler not loaded"}
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    # Calculate stress based on mobile usage (higher usage = higher stress, simplified formula)
    mobile_usage = data.get('mobile_usage', 0)
    stress_level = min(100, mobile_usage / 2)  # Scale 0-12 hours to 0-100 stress level
    patient_df = pd.DataFrame([{
        'TotalSteps': data.get('steps', 0),
        'TotalDistance': data.get('distance', 0),
        'VeryActiveMinutes': data.get('very_active_minutes', 0),
        'FairlyActiveMinutes': data.get('fairly_active_minutes', 0),
        'LightlyActiveMinutes': data.get('lightly_active_minutes', 0),
        'SedentaryMinutes': data.get('sedentary_minutes', 0),
        'WeightKg': data.get('weight', 0),
        'BMI': data.get('bmi', 0),
        'HeartRate': data.get('heart_rate', 0),
        'MobileUsage': mobile_usage,
        'WaterIntake': data.get('water_intake', 0),
        'SystolicBP': data.get('systolic_bp', 0),
        'DiastolicBP': data.get('diastolic_bp', 0)
    }])
    patient_df = patient_df.fillna(patient_df.median())
    X_patient_scaled = scaler.transform(patient_df)
    calories = rf_model.predict(X_patient_scaled)[0]
    data['calories'] = calories
    c.execute("INSERT INTO health_data (id, timestamp, steps, distance, calories, weight, bmi, heart_rate, mobile_usage, water_intake, systolic_bp, diastolic_bp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (data.get('id', 0), datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data.get('steps', 0), data.get('distance', 0),
               calories, data.get('weight', 0), data.get('bmi', 0), data.get('heart_rate', 0), mobile_usage,
               data.get('water_intake', 0), data.get('systolic_bp', 0), data.get('diastolic_bp', 0)))
    conn.commit()
    conn.close()
    socketio.emit('sensor_data', data)
    print(f"Processed data: {data} at {datetime.now()}")
    # Health details summary
    health_details = {
        "Steps": data.get('steps', 0),
        "Distance (km)": data.get('distance', 0),
        "Calories": calories,
        "Weight (kg)": data.get('weight', 0),
        "BMI": data.get('bmi', 0),
        "Heart Rate (bpm)": data.get('heart_rate', 0),
        "Mobile Usage (hours)": mobile_usage,
        "Stress Level (%)": stress_level,
        "Water Intake (liters)": data.get('water_intake', 0),
        "Blood Pressure (mmHg)": f"{data.get('systolic_bp', 0)}/{data.get('diastolic_bp', 0)}"
    }
    return {"message": "Data processed", "calories": calories, "health_details": health_details}

# API to get health recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    steps = data.get('steps', 0)
    calories = data.get('calories', 0)
    bmi = data.get('bmi', 0)
    heart_rate = data.get('heart_rate', 0)
    stress_level = min(100, data.get('mobile_usage', 0) / 2)
    water_intake = data.get('water_intake', 0)
    systolic_bp = data.get('systolic_bp', 0)
    diastolic_bp = data.get('diastolic_bp', 0)
    recommendations = []
    if steps < 5000:
        recommendations.append("Increase daily steps to at least 10,000 for better health.")
    if bmi > 24.9:
        recommendations.append("Consider a balanced diet to manage BMI.")
    if calories < 2000:
        recommendations.append("Ensure adequate calorie intake for energy.")
    if heart_rate > 100:
        recommendations.append("Monitor heart rate; consult a doctor if consistently high.")
    if stress_level > 50:
        recommendations.append("Reduce screen time and consider stress-relief exercises.")
    if water_intake < 1.5:
        recommendations.append("Aim to drink at least 1.5-2 liters of water daily.")
    if systolic_bp > 130 or diastolic_bp > 90:
        recommendations.append("Check blood pressure regularly; consult a healthcare provider if elevated.")
    return jsonify({"recommendations": recommendations})

# Handle real-time data updates
@app.route('/update_data', methods=['POST'])
def update_data():
    print(f"Received /update_data request at {datetime.now()}")
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.json
    print(f"Received data: {data} at {datetime.now()}")
    result = process_health_data(data)
    if "error" in result:
        return jsonify(result), 500
    # Fetch recommendations immediately
    rec_result = recommend()
    result.update(rec_result.get_json())
    return jsonify(result), 200

# Test route to verify server
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 200

# Serve the root URL with index.html
@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('.', 'index.html')

# Handle favicon request
@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return '', 204

if __name__ == '__main__':
    print(f"Starting server at {datetime.now()}")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)