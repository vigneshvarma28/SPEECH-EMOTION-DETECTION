import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
from decoder import decode_features
import pandas as pd

def train_lstm_model(features, labels):
    le = LabelEncoder()
    labels_encoded = to_categorical(le.fit_transform(labels))
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(labels_encoded.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels, target_names=le.classes_))
    
    print("Final validation accuracy:", history.history['val_accuracy'][-1])
    
    return model, le, scaler

def load_model():
    features, labels = decode_features('features.csv')
    model, le, scaler = train_lstm_model(features, labels)
    return model, le, scaler

if __name__ == "__main__":
    model, label_encoder, scaler = load_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('emotion_model.h5')
    np.save('label_encoder.npy', label_encoder.classes_)
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
    
    # Add feature analysis here
    df = pd.read_csv('features.csv')
    print("Feature statistics from training data:")
    print(df.drop(columns=['emotion']).describe())