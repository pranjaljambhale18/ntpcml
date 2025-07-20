import streamlit as st
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the model and feature list
model = joblib.load("ntpc_model.pkl")
feature_columns = joblib.load("model_features.pkl")  # List of training features

# Define default values and ranges for sliders (customize as per your dataset)
default_ranges = {
    "PLF": (30, 100, 75),
    "Coal_Consumption": (2000, 7000, 4500),
    "Aux_Consumption": (10, 40, 22),
    "Gross_Generation": (0, 10000, 5000),
    "Net_Generation": (0, 9500, 4500),
    "Steam_Pressure": (30, 200, 120),
    "Steam_Temperature": (300, 600, 500),
    # Add more features with suitable ranges
}

st.set_page_config(page_title="NTPC Predictor", layout="centered")
st.title("⚙️ NTPC Multi-Output Predictor")
st.markdown("Adjust the input values below to get predictions and suggestions.")

# User input form
input_data = {}
with st.form("prediction_form"):
    for feature in feature_columns:
        if feature in default_ranges:
            min_val, max_val, default = default_ranges[feature]
            input_data[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=default)
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

    email = st.text_input("📧 Enter your email to receive predictions (optional)", "")
    submitted = st.form_submit_button("🔍 Predict")

# Handle prediction
if submitted:
    try:
        input_df = pd.DataFrame([input_data])[feature_columns]
        prediction = model.predict(input_df)[0]

        st.success("✅ Prediction successful!")
        st.subheader("🔢 Predicted Outputs:")
        prediction_df = pd.DataFrame(prediction.reshape(1, -1), columns=[f"Output {i+1}" for i in range(len(prediction))])
        st.dataframe(prediction_df)

        # Generate suggestions
        st.subheader("📌 Suggestions:")
        suggestions = []
        if input_data["PLF"] < 60:
            suggestions.append("⚠️ Increase PLF above 60% for better performance.")
        if input_data.get("Coal_Consumption", 0) > 5000:
            suggestions.append("⚠️ Try to reduce Coal Consumption below 5000 units.")
        if input_data.get("Aux_Consumption", 0) > 25:
            suggestions.append("⚠️ Optimize auxiliary consumption to be below 25 units.")
        if not suggestions:
            suggestions.append("✅ All inputs are within optimal range.")

        for suggestion in suggestions:
            st.write(suggestion)

        # Email prediction
        if email:
            try:
                sender_email = "your_email@example.com"
                sender_password = "your_password"
                receiver_email = email

                message = MIMEMultipart("alternative")
                message["Subject"] = "NTPC Prediction Results"
                message["From"] = sender_email
                message["To"] = receiver_email

                html = f"""
                <html>
                <body>
                    <h2>NTPC Prediction Results</h2>
                    <p><b>Predictions:</b> {prediction.tolist()}</p>
                    <p><b>Suggestions:</b><ul>{"".join(f"<li>{s}</li>" for s in suggestions)}</ul></p>
                </body>
                </html>
                """
                message.attach(MIMEText(html, "html"))

                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.sendmail(sender_email, receiver_email, message.as_string())

                st.success("📩 Email sent successfully!")

            except Exception as e:
                st.error(f"Email failed: {e}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
