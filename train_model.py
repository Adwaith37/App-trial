import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load data
file_path = 'dummy_npi_data.xlsx'
df = pd.read_excel(file_path)

# Rename columns for consistency
df.rename(columns={
    'Login Time': 'login_time',
    'Usage Time (mins)': 'usage_time',
    'Count of Survey Attempts': 'survey_attempts',
    'State': 'state',
    'Region': 'region',
    'Speciality': 'specialty'
}, inplace=True)

# Convert login_time to datetime
df['login_time'] = pd.to_datetime(df['login_time'])

# Extract day of the week and encode it
all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_encoder = LabelEncoder()
day_encoder.fit(all_days)  # Fit on all days to avoid unseen label issues
df['day_of_week'] = day_encoder.transform(df['login_time'].dt.day_name())

# Encode categorical features
state_encoder = LabelEncoder()
region_encoder = LabelEncoder()
specialty_encoder = LabelEncoder()

df['state_encoded'] = state_encoder.fit_transform(df['state'])
df['region_encoded'] = region_encoder.fit_transform(df['region'])
df['specialty_encoded'] = specialty_encoder.fit_transform(df['specialty'])

# Define target (1 if survey_attempts > 0, else 0)
df['survey_attempted'] = (df['survey_attempts'] > 0).astype(int)

# Define features and target
X = df[['state_encoded', 'region_encoded', 'specialty_encoded', 'day_of_week', 'usage_time']]
y = df['survey_attempted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')

# Save model and encoders
joblib.dump(model, 'doctor_survey_model.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(region_encoder, 'region_encoder.pkl')
joblib.dump(specialty_encoder, 'specialty_encoder.pkl')
joblib.dump(day_encoder, 'day_encoder.pkl')

print("Model and encoders saved successfully!")




