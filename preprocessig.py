import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Define paths to dataset folders
folder_paths = [
    'mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16',
    'mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 3.12.16-4.11.16'
]

# List of relevant CSV files to merge
data_files = [
    'dailyActivity_merged.csv', 'hourlyCalories_merged.csv', 'hourlyIntensities_merged.csv',
    'hourlySteps_merged.csv', 'minuteCaloriesNarrow_merged.csv', 'minuteIntensitiesNarrow_merged.csv',
    'minuteMETsNarrow_merged.csv', 'minuteSleep_merged.csv', 'minuteStepsNarrow_merged.csv',
    'sleepDay_merged.csv', 'weightLogInfo_merged.csv'
]

# Initialize list to store dataframes
df_list = []

# Load and merge data from both folders
for folder_path in folder_paths:
    for file in data_files:
        full_path = os.path.join(folder_path, file)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            df_list.append(df)

# Concatenate all dataframes
df_merged = pd.concat(df_list, ignore_index=True)

# Define date formats for each column
date_formats = {
    'ActivityDate': '%m/%d/%Y',  # e.g., 3/25/2016
    'Date': '%m/%d/%Y',         # e.g., 3/12/2016
    'ActivityHour': '%m/%d/%Y %I:%M:%S %p',  # e.g., 3/12/2016 12:00:00 AM
    'date': '%m/%d/%Y %I:%M:%S %p'           # e.g., 3/13/2016 2:39:30 AM
}

# Preprocess date columns with specific formats
date_cols = ['ActivityDate', 'Date', 'ActivityHour', 'date']
for col in date_cols:
    if col in df_merged.columns:
        try:
            df_merged[col] = pd.to_datetime(df_merged[col], format=date_formats.get(col, None), errors='coerce')
        except ValueError as e:
            print(f"Warning: Could not parse {col} with format {date_formats.get(col, 'default')}. Falling back to dateutil. Error: {e}")
            df_merged[col] = pd.to_datetime(df_merged[col], errors='coerce')

# Handle numeric columns and missing values
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
df_merged[numeric_cols] = df_merged[numeric_cols].fillna(df_merged[numeric_cols].median())

# Map boolean-like columns
df_merged['IsManualReport'] = df_merged['IsManualReport'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0)

# Drop irrelevant or highly missing columns
df_merged = df_merged.drop(columns=['Fat', 'LogId', 'logId', 'value', 'ActivityMinute', 'Calories_minuteCaloriesNarrow'], errors='ignore')

# Features and target
features = ['TotalSteps', 'TotalDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 'WeightKg', 'BMI', 'Calories']
df_merged = df_merged.dropna(subset=features)  # Drop rows with missing values in key features
X = df_merged[features[:-1]]  # Exclude Calories as target
y = df_merged['Calories']

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
rf_model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
rf_model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(rf_model.best_estimator_, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")

# Optional: Save merged dataset for inspection
df_merged.to_csv('merged_fitbit_dataset.csv', index=False)
print("Merged dataset saved as 'merged_fitbit_dataset.csv'.")