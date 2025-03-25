import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Load the dataset
df=pd.read_csv('/home/jmal/Desktop/tp/machine_learning/Crop_recommendation.csv')

features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']


# Encode the target variable
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
joblib.dump(label_encoder, 'label_encoder.joblib')
# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform k-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_classifier, features, target_encoded, cv=kf)

# Print cross-validation scores
print(f"K-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean K-Fold Cross-Validation Score: {cv_scores.mean():.2f}")

# Train the model on the entire dataset
rf_classifier.fit(features, target_encoded)

# Save the trained model using joblib
joblib.dump(rf_classifier, 'random_forest_model.joblib')

# Function to recommend the most suitable crop for given soil features
def recommend_crop(soil_features, model, label_encoder):
    soil_features = np.array(soil_features).reshape(1, -1)
    predicted_proba = model.predict_proba(soil_features)[0]
    best_crop_index = np.argmax(predicted_proba)
    best_crop = label_encoder.inverse_transform([best_crop_index])[0]
    return best_crop, predicted_proba[best_crop_index]

# Load the model from the file
loaded_model = joblib.load('random_forest_model.joblib')

# Example soil features
soil_features = [70, 30, 40, 25, 70, 6.5, 150]  # Example values for N, P, K, temperature, humidity, ph, rainfall

# Get the most suitable crop recommendation
best_crop, probability = recommend_crop(soil_features, loaded_model, label_encoder)

print(f"The most suitable crop for the given soil conditions is: {best_crop}")
print(f"Probability of suitability: {probability:.2f}")