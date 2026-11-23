
# ==========================================
# CROP RECOMMENDATION SYSTEM
# Dataset Loading and Exploratory Data Analysis
# ==========================================

# Step 1: Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load dataset

data = pd.read_csv("Crop_recommendation.csv")

print("\n========== FIRST 5 ROWS ==========")
print(data.head())


# Step 3: Dataset dimensions

print("\n========== DATASET SHAPE ==========")
print(data.shape)


# Step 4: Dataset information

print("\n========== DATASET INFORMATION ==========")
data.info()


# Step 5: Check missing values

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())


# Step 6: Crop distribution

print("\n========== CROP DISTRIBUTION ==========")
print(data["label"].value_counts())


# Step 7: Statistical summary

print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())


# Step 8: Unique crop categories

print("\n========== UNIQUE CROPS ==========")
print(data["label"].unique())

print("\nTotal crop categories:", data["label"].nunique())


# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================


# Step 9: Nitrogen distribution histogram

plt.figure(figsize=(8, 5))

sns.histplot(data["N"], bins=20, kde=True)

plt.title("Nitrogen Distribution")
plt.xlabel("Nitrogen Level")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Step 10: Nitrogen boxplot for different crops

plt.figure(figsize=(14, 6))

sns.boxplot(data=data, x="label", y="N")

plt.title("Nitrogen Distribution Across Different Crops")
plt.xlabel("Crop")
plt.ylabel("Nitrogen Level")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Step 11: Correlation heatmap

# Select numerical columns only
numerical_data = data.select_dtypes(include="number")

correlation = numerical_data.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()


print("\n========== EDA COMPLETED SUCCESSFULLY ==========")

# ==========================================
# STEP 12: FEATURE AND TARGET SEPARATION
# ==========================================

X = data.drop("label", axis=1)
y = data["label"]

print("\nInput Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())


# ==========================================
# STEP 13: TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)

print("\nDATA PREPARATION COMPLETED!")

# ==========================================
# STEP 14: RANDOM FOREST MODEL TRAINING
# ==========================================

from sklearn.ensemble import RandomForestClassifier

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED!")


# ==========================================
# STEP 15: MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

print("\nFirst 10 Actual Crops:")
print(y_test.head(10).values)

print("\nFirst 10 Predicted Crops:")
print(y_pred[:10])

# ==========================================
# STEP 16: MODEL EVALUATION
# ==========================================

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(accuracy * 100, "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# ==========================================
# STEP 17: CONFUSION MATRIX
# ==========================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(14, 12))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(
    cmap="Blues",
    xticks_rotation=90,
    values_format="d"
)

plt.title("Crop Recommendation - Confusion Matrix")
plt.tight_layout()
plt.show()


# ==========================================
# STEP 18: SAVE TRAINED MODEL
# ==========================================

import pickle

with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nMODEL SAVED SUCCESSFULLY!")

