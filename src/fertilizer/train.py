
"""
Fertilizer Recommendation System
Model Training File
Algorithm: Random Forest Classifier
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Project directory
ROOT = Path(__file__).resolve().parent

# Dataset and model paths
DATA_PATH = ROOT / "fertilizer_dataset.csv"
MODEL_PATH = ROOT / "fertilizer_model.joblib"


# Input features
NUMERIC = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "Moisture"
]

CATEGORICAL = [
    "Soil_Type",
    "Crop_Type"
]

FEATURES = NUMERIC + CATEGORICAL
TARGET = "Fertilizer"


def main():

    # Load dataset
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Data preprocessing
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL)
    ])

    # Random Forest model
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Model prediction on test data
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    print("Model Training Completed!")
    print("Test Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save trained model and metadata
    artifact = {
        "model": model,
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "classes": sorted(y.unique().tolist()),
        "dataset_note": "Synthetic educational data; not validated for farm use."
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Model location:", MODEL_PATH)


if __name__ == "__main__":
    main()
