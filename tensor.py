import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

# Example dataset
df=pd.read_csv("/home/jmal/Desktop/tp/machine_learning/Crop_recommendation.csv")

# Define features and target variable
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Encode the target variable
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Train the scikit-learn model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(features, target_encoded)

# Define a simple neural network
input_shape = (7,)  # Adjust based on your feature count
num_classes = len(label_encoder.classes_)

# Create a TensorFlow model
tf_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the TensorFlow model using the same dataset
tf_model.fit(features, target_encoded, epochs=20, batch_size=1)

# Save the TensorFlow model with the correct extension
tf_model.save('tf_model.keras')

# Save the LabelEncoder using pickle
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

# Convert the TensorFlow model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()



# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model and LabelEncoder saved successfully.")