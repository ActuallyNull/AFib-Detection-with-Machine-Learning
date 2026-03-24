from focal_loss import SparseCategoricalFocalLoss
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import wfdb
import os
from scipy.signal import butter, filtfilt
import neurokit2 as nk
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model, scaler, and label encoder
model = keras.models.load_model(
    "Multimodal_CNN_Test/multimodal_model.keras",
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
)
scaler = joblib.load("Multimodal_CNN_Test/scaler.pkl")
le = LabelEncoder()
le.classes_ = np.load("Multimodal_CNN_Test/le.npy", allow_pickle=True)

# --- Configuration ---
INPUT_LENGTH = 3000  # <-- change to the length your model expects

def butter_bandpass_filter(signal, fs, low_pass=0.5, high_pass=50, order=5):
    nyq = 0.5 * fs
    low = low_pass / nyq
    high = high_pass / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_features(ecg_signals, fs):
    all_features = []
    for signal in ecg_signals:
        try:
            # Process the signal
            signals, info = nk.ecg_process(signal, sampling_rate=fs)
            # Extract features
            features = nk.ecg_intervalrelated(signals, sampling_rate=fs)
            all_features.append(features)
        except Exception as e:
            # If feature extraction fails, append a DataFrame of NaNs
            # This is to maintain the structure and handle it later
            print(f"Feature extraction failed for a signal: {e}")
            all_features.append(pd.DataFrame(np.nan, index=[0], columns=all_features[-1].columns if all_features else []))
    
    # Concatenate all feature DataFrames
    features_df = pd.concat(all_features, ignore_index=True)
    # Replace inf values with NaN
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaNs with the mean of the column
    features_df.fillna(features_df.mean(), inplace=True)
    # Fill any remaining NaNs with 0 (in case a whole column is NaN)
    features_df.fillna(0, inplace=True)
    return features_df

def load_ecg(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load ECG data from either CSV or WFDB format.
    Returns a 1D numpy array of the ECG signal and the sampling frequency.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    fs = 300 # default sampling frequency

    if ext == ".csv":
        df = pd.read_csv(file_path, header=None)
        signal = df.values.flatten()

    elif ext == ".dat" or ext == ".hea":
        # Strip extension for wfdb
        record_name = file_path.replace(ext, "")
        record = wfdb.rdrecord(record_name)
        # Use the first channel by default
        signal = record.p_signal[:, 0]
        fs = record.fs

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return signal, fs

def preprocess_ecg(signal: np.ndarray, fs: int, features, scaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess ECG signal and features into model-ready input.
    """
    # 1. Filter the signal
    signal = butter_bandpass_filter(signal, fs)

    # 2. Pad or truncate
    if len(signal) < INPUT_LENGTH:
        signal = np.pad(signal, (0, INPUT_LENGTH - len(signal)))
    else:
        signal = signal[:INPUT_LENGTH]

    # 3. Normalize (min-max)
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)

    # 4. Add batch + channel dims → (1, length, 1)
    signal = np.expand_dims(signal, axis=(0, -1)).astype(np.float32)

    # 5. Scale features
    scaled_features = scaler.transform(features)

    return signal, scaled_features

def classify_ecg(file_path: str):
    """
    Run model prediction on an ECG file (CSV or WFDB).
    """
    raw_signal, fs = load_ecg(file_path)
    features = extract_features([raw_signal], fs)
    processed_signal, scaled_features = preprocess_ecg(raw_signal, fs, features, scaler)

    preds = model.predict([processed_signal, scaled_features])
    pred_class_idx = np.argmax(preds, axis=1)[0]
    pred_class_name = le.inverse_transform([pred_class_idx])[0]
    confidence = float(np.max(tf.nn.softmax(preds, axis=1)))

    return {
        "predicted_class": pred_class_name,
        "confidence": confidence,
        "raw_logits": preds.tolist()
    }

base_file_path = "training2017/training2017/"

if __name__ == "__main__":

    while True:
        test_file = input("Enter the path to the ECG file: ")
        result = classify_ecg(base_file_path + test_file)
        print("Prediction:", result)
