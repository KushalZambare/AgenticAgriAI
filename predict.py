
import pickle
import pandas as pd


# ==========================================
# 1. INPUT VALIDATION FUNCTION
# ==========================================

def get_valid_input(message, minimum, maximum):
    while True:
        try:
            value = float(input(message))

            if not minimum <= value <= maximum:
                print(f"Please enter a value between {minimum} and {maximum}.")
                continue

            return value

        except ValueError:
            print("Invalid input! Please enter a numeric value.")


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

try:
    with open("crop_model.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    print("Error: crop_model.pkl not found!")
    print("Please run train_model.py first.")
    exit()


# ==========================================
# 3. DISPLAY SYSTEM HEADING
# ==========================================

print("\n========================================")
print("       CROP RECOMMENDATION SYSTEM")
print("========================================")
print("Enter your soil and weather information.\n")


# ==========================================
# 4. TAKE INPUT WITH VALIDATION
# ==========================================

N = get_valid_input("Nitrogen (N): ", 0, 140)
P = get_valid_input("Phosphorus (P): ", 5, 145)
K = get_valid_input("Potassium (K): ", 5, 205)

temperature = get_valid_input("Temperature (C): ", 8, 44)
humidity = get_valid_input("Humidity (%): ", 14, 100)
ph = get_valid_input("Soil pH: ", 3.5, 10)
rainfall = get_valid_input("Rainfall (mm): ", 20, 300)


# ==========================================
# 5. PREPARE INPUT DATA
# ==========================================

input_data = pd.DataFrame([{
    "N": N,
    "P": P,
    "K": K,
    "temperature": temperature,
    "humidity": humidity,
    "ph": ph,
    "rainfall": rainfall
}])


# ==========================================
# 6. PREDICT THE CROP
# ==========================================

prediction = model.predict(input_data)

probabilities = model.predict_proba(input_data)[0]


# ==========================================
# 7. DISPLAY RECOMMENDATION
# ==========================================

print("\n========================================")
print("          RECOMMENDATION")
print("========================================")

print(f"\nRecommended Crop: {prediction[0].upper()}")


# ==========================================
# 8. DISPLAY TOP 3 CROPS
# ==========================================

top_3 = probabilities.argsort()[-3:][::-1]

print("\nTop 3 Crop Recommendations:")
print("----------------------------------------")

for rank, index in enumerate(top_3, start=1):
    crop = model.classes_[index]
    score = probabilities[index] * 100

    print(f"{rank}. {crop.capitalize()} - {score:.2f}%")


# ==========================================
# 9. DISPLAY INPUT SUMMARY
# ==========================================

print("\n========================================")
print("             INPUT SUMMARY")
print("========================================")

print(f"Nitrogen       : {N}")
print(f"Phosphorus     : {P}")
print(f"Potassium      : {K}")
print(f"Temperature    : {temperature} C")
print(f"Humidity       : {humidity}%")
print(f"Soil pH        : {ph}")
print(f"Rainfall       : {rainfall} mm")

print("\n========================================")
print("       PREDICTION COMPLETED!")
print("========================================")

