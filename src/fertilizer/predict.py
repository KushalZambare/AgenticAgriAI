
"""
Fertilizer Recommendation System
Prediction File
"""

from pathlib import Path
import argparse
import joblib
import pandas as pd


# Load trained model
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fertilizer_model.joblib"


def get_input():

    print("\n===== FERTILIZER RECOMMENDATION SYSTEM =====")

    nitrogen = float(input("Nitrogen value (N): "))
    phosphorus = float(input("Phosphorus value (P): "))
    potassium = float(input("Potassium value (K): "))

    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    moisture = float(input("Soil moisture (%): "))

    soil_type = input(
        "Soil type (Sandy/Loamy/Black/Clayey/Red): "
    ).strip()

    crop_type = input(
        "Crop (Rice/Wheat/Maize/Cotton/Sugarcane/Groundnut/Tomato): "
    ).strip()

    return {
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Temperature": temperature,
        "Humidity": humidity,
        "Moisture": moisture,
        "Soil_Type": soil_type,
        "Crop_Type": crop_type
    }


def predict_fertilizer():

    # Check model availability
    if not MODEL_PATH.exists():
        print("Model not found. Please run train.py first.")
        return

    # Load model artifact
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]

    # Get user input
    values = get_input()

    # Convert input into DataFrame
    input_data = pd.DataFrame(
        [values],
        columns=artifact["features"]
    )

    # Predict fertilizer
    prediction = model.predict(input_data)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    print("\n===== FERTILIZER RECOMMENDATION =====")

    print("Recommended Fertilizer:", prediction)

    print("Model Confidence: {:.2f}%".format(confidence))

    # Display top 3 predictions
    print("\nTop 3 Predictions:")

    ranked = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    for fertilizer, probability in ranked:

        print(
            "{} : {:.2f}%".format(
                fertilizer,
                probability * 100
            )
        )

    print("\nNote: This is an educational ML prototype.")
    print("Use certified soil testing and expert guidance")
    print("for actual fertilizer selection and dosage.")


if __name__ == "__main__":
    predict_fertilizer()
