from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load trained model and scaler
model = tf.keras.models.load_model("sepsis_model.h5")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form submission
        data = request.form.to_dict()
        
        # Convert input values to float
        input_values = np.array([float(data[key]) for key in data.keys()]).reshape(1, -1)

        # Scale input data
        input_scaled = scaler.transform(input_values)

        # Make prediction
        prediction = model.predict(input_scaled)[0][0]
        result = "Positive" if prediction > 0.5 else "Negative"

        return jsonify({"Sepsis Prediction": result, "Confidence": f"{prediction*100:.2f}%"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
