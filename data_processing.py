import wfdb
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import SMOTE
from data_augmentor import ECG_Augmentor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc

# ====================
# Signal Processing
# ====================
def butter_bandpass_filter(signal, fs, low_pass=0.5, high_pass=50, order=5):
    nyq = 0.5 * fs
    low = low_pass / nyq
    high = high_pass / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def load_physionet_dataset(physionet_path, reference_file_path, duration=10):
    physionet_label_map = {'N': "Normal", 'A': "AFib", 'O': "Other Arrythmia"}
    reference_df = pd.read_csv(reference_file_path, header=None, names=['record', 'label'])

    X, y = [], []
    for file in os.listdir(physionet_path):
        if file.endswith(".hea"):
            record_name = file[:-4]
            label_row = reference_df.loc[reference_df['record'] == record_name]
            if label_row.empty:
                continue
            label_code = label_row['label'].values[0]
            if label_code == "~":  # Skip noisy
                continue
            label_str = physionet_label_map.get(label_code, "Other Arrythmia")

            record = wfdb.rdrecord(os.path.join(physionet_path, record_name))
            fs = record.fs
            samples = fs * duration
            ecg_signal = record.p_signal[:, 0]
            trunc_ecg = ecg_signal[:samples]
            if len(trunc_ecg) < samples:
                trunc_ecg = np.pad(trunc_ecg, (0, samples - len(trunc_ecg)), 'constant')

            X.append(trunc_ecg)
            y.append(label_str)

    return np.array(X, dtype=np.float32), np.array(y)

# ====================
# Preprocessing
# ====================
def preprocess_dataset(physionet_path, reference_file_path, fs=300):
    X, y = load_physionet_dataset(physionet_path, reference_file_path)
    X_filtered = np.array([butter_bandpass_filter(x, fs=fs) for x in X], dtype=np.float32)
    return X_filtered, y

# ====================
# Train-Val-Test Split
# ====================
def create_train_val_test_splits(fs=300):
    augmentor = ECG_Augmentor(fs=fs)

    # Load & filter
    X_filtered, y = preprocess_dataset(
        "training2017/training2017",
        "training2017/training2017/REFERENCE.csv",
        fs=fs
    )

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split first (avoid leakage)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_filtered, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Fit scaler only on train
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Augment training data
    print(f"data size before augmentation: {X_train.shape}")
    batch_size = 500
    X_train_aug = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i+batch_size]
        batch_aug = [augmentor.augment(x) for x in batch]
        X_train_aug.append(np.array(batch_aug, dtype=np.float32))
    X_train_aug = np.concatenate(X_train_aug, axis=0)
    print(f"data size after augmentation: {X_train_aug.shape}")

    # Optional: SMOTE only on train (if needed)
    # smote = SMOTE(random_state=42)
    # X_train_aug, y_train = smote.fit_resample(X_train_aug, y_train)

    # Reshape for CNN
    X_train_aug = np.expand_dims(X_train_aug, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    gc.collect()
    return X_train_aug, y_train, X_val, y_val, X_test, y_test, le