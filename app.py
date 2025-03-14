import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import io

# Load the trained model and encoders
model = joblib.load('doctor_survey_model.pkl')
state_encoder = joblib.load('state_encoder.pkl')
region_encoder = joblib.load('region_encoder.pkl')
specialty_encoder = joblib.load('specialty_encoder.pkl')
day_encoder = joblib.load('day_encoder.pkl')

# Streamlit UI
st.title('Doctor Survey Prediction App')

# Input fields
state = st.selectbox('Select State', state_encoder.classes_)
region = st.selectbox('Select Region', region_encoder.classes_)
specialty = st.selectbox('Select Specialty', specialty_encoder.classes_)
usage_time = st.number_input('Usage Time (in minutes)', min_value=0.0, format="%.2f")

# Store predictions for CSV
predictions = []

if st.button('Predict'):
    # Get today's day of the week and map to encoding
    day_of_week_str = datetime.today().strftime('%A')
    
    if day_of_week_str not in day_encoder.classes_:
        st.error(f"Day '{day_of_week_str}' not recognized. Try using a known day.")
    else:
        day_of_week = day_encoder.transform([day_of_week_str])[0]

        # Encode input values
        state_encoded = state_encoder.transform([state])[0]
        region_encoded = region_encoder.transform([region])[0]
        specialty_encoded = specialty_encoder.transform([specialty])[0]

        # Create input data for prediction
        input_data = [[state_encoded, region_encoded, specialty_encoded, day_of_week, usage_time]]

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = 'LIKELY' if prediction == 1 else 'UNLIKELY'

        # Display result
        st.write(f"**Prediction:** The doctor is **{result}** to attempt the survey.")

        # Save prediction to list
        predictions.append({
            'State': state,
            'Region': region,
            'Specialty': specialty,
            'Day of Week': day_of_week_str,
            'Usage Time': usage_time,
            'Prediction': result
        })

# Download CSV Section
if predictions:
    # Convert predictions to DataFrame
    df = pd.DataFrame(predictions)

    # Create CSV buffer
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Download button
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_buffer.getvalue(),
        file_name="doctor_survey_predictions.csv",
        mime="text/csv"
    )








