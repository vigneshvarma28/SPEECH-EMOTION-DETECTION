from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from extract_features import extract_features
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:8000"}})

# Load pre-trained model, label encoder, and scaler
model = keras_load_model('emotion_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy', allow_pickle=True)
scaler = joblib.load('scaler.pkl')  # Load the scaler
print("Label encoder classes:", label_encoder.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received predict request") 
    audio_file = request.files['audio']
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)
    print(f"Saved audio to {audio_path}")
    
    try:
        feature_dict = extract_features(audio_path)
        if feature_dict is None:
            raise ValueError("Feature extraction returned None")
        print("Features extracted:", feature_dict)
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        # Keep the file for debugging
        # os.remove(audio_path)
        return jsonify({'error': f'Failed to extract features: {str(e)}'}), 400
    
    feature_dict.pop('emotion', None)  # Remove 'emotion' if present
    feature_vector = [list(feature_dict.values())]
    feature_vector = np.array(feature_vector)
    expected_features = 20
    if feature_vector.shape[1] != expected_features:
        os.remove(audio_path)
        print(f"Feature mismatch: expected {expected_features}, got {feature_vector.shape[1]}")
        return jsonify({'error': f'Feature mismatch: expected {expected_features}, got {feature_vector.shape[1]}'}), 400
    
    # Normalize the features
    feature_vector = scaler.transform(feature_vector)
    
    feature_vector = feature_vector.reshape((1, 1, feature_vector.shape[1]))
    prediction = model.predict(feature_vector)
    print("Raw prediction probabilities:", prediction)
    emotion_idx = np.argmax(prediction)
    emotion = label_encoder.inverse_transform([emotion_idx])[0]
    print(f"Predicted emotion: {emotion}")
    
    os.remove(audio_path)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True, port=5000)