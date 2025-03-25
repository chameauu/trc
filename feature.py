import pandas as pd
import joblib
import numpy as np
from scipy.optimize import minimize

new_df = pd.read_csv('/home/jmal/Desktop/tp/machine_learning/Random_Crop_Recommendation.csv')  # Replace with the path to your new dataset

# Define features for the new dataset
features = new_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Load the label encoder used during training
label_encoder = joblib.load('label_encoder.joblib')  # Ensure you saved the label encoder during training

def predict_probability(conditions, model, crop_label):
    conditions = np.array(conditions).reshape(1, -1)
    probabilities = model.predict_proba(conditions)
    return probabilities[0][crop_label]

# Function to optimize soil conditions for a given crop
def optimize_soil_conditions(crop_name, model, label_encoder, feature_names):
    crop_label = label_encoder.transform([crop_name])[0]

    # Initial guess for soil conditions (mean values)
    initial_conditions = features.mean().values

    # Define bounds for each feature
    bounds = [(features[col].min(), features[col].max()) for col in feature_names]

    # Minimize the negative probability (equivalent to maximizing the probability)
    result = minimize(
        lambda conditions: -predict_probability(conditions, model, crop_label),
        initial_conditions,
        bounds=bounds
    )

    # Extract the optimal conditions
    optimal_conditions = dict(zip(feature_names, result.x))
    return optimal_conditions

crop_name = 'rice'
optimal_conditions = optimize_soil_conditions(crop_name, model, label_encoder, features.columns)

print(f"Optimal soil conditions for {crop_name}:")
print(optimal_conditions)