import pandas as pd
import joblib
import numpy as np

df = pd.read_csv('/home/jmal/Desktop/tp/machine_learning/Crop_recommendation.csv')

# Define features and target variable
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

loaded_model = joblib.load('random_forest_model.joblib')

# Load the label encoder used during training
label_encoder = joblib.load('label_encoder.joblib')  # Ensure you saved the label encoder during training


def get_soil_enhancement_recommendations(soil_features, crop_name, model, label_encoder, feature_names):
    crop_label = label_encoder.transform([crop_name])[0]
    soil_features = np.array(soil_features).reshape(1, -1)

    # Predict the probability of the crop
    probabilities = model.predict_proba(soil_features)[0]
    crop_probability = probabilities[crop_label]

    # Get feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances,
        'Current Value': soil_features[0]
    })

    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Recommendations based on feature importance and current values
    recommendations = {}
    for _, row in importance_df.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        current_value = row['Current Value']

        # Provide a simple recommendation based on importance and current value
        if importance > 0:
            if current_value < df[feature].mean():
                recommendations[feature] = f"Increase {feature}"
            else:
                recommendations[feature] = f"Maintain or slightly adjust {feature}"

    return recommendations, crop_probability

soil_features = [9  0, 30, 40, 50, 70, 6.5, 150]  # Example values for N, P, K, temperature, humidity, ph, rainfall
crop_name = 'banana'

# Get recommendations
recommendations, crop_probability = get_soil_enhancement_recommendations(soil_features, crop_name, loaded_model, label_encoder, features.columns)

print(f"Recommendations for enhancing soil for {crop_name}:")
print(recommendations)