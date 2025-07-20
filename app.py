import streamlit as st
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the model and expected features
model = joblib.load("ntpc_model.pkl")
feature_columns = joblib.load("model_features.pkl")  # list of training feature names

st.set_page_config(page_title="NTPC Multi-Output Predictor", layout="centered")

st.title("⚙️ NTPC Multi-Output Predictor")
st.markdown("Enter the input values below to get predictions and suggestions.")

# Dynamic input form based on expected features
input_data = {}
with st.form("prediction_form"):
    for feature in feature_columns:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
    email = st.text_input("📧 Enter your email to receive predictions (optional)", "")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_df = pd.DataFrame([input_data])[feature_columns]  # correct order
        prediction = model.predict(input_df)[0]

        st.success("✅ Prediction successful!")
        st.subheader("🔢 Predicted Outputs:")
        prediction_df = pd.DataFrame(prediction.reshape(1, -1), columns=[f"Output {i+1}" for i in range(len(prediction))])
        st.dataframe(prediction_df)

        # Suggestion logic (basic — adjust per your use case)
        st.subheader("📌 Suggestions:")
        suggestions = []
        if input_data["PLF"] < 60:
            suggestions.append("Increase PLF above 60% for better performance.")
        if input_data.get("Coal_Consumption", 0) > 5000:
            suggestions.append("Try to reduce Coal Consumption below 5000 units.")
        if input_data.get("Aux_Consumption", 0) > 25:
            suggestions.append("Optimize auxiliary consumption to be below 25 units.")
        if not suggestions:
            suggestions.append("All inputs are within optimal range.")

        for suggestion in suggestions:
            st.write(f"✅ {suggestion}")

        # Email integration
        if email:
            try:
                sender_email = "your_email@example.com"  # replace with sender
                sender_password = "your_password"         # replace with password/app password
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

                st.success("📧 Email sent successfully!")

            except Exception as e:
                st.error(f"Failed to send email: {e}")

    except ValueError as e:
        st.error(f"Input error: {e}")
    except Exception as ex:
        st.error(f"❌ Prediction failed: {ex}")
