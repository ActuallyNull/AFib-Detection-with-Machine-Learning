import wfdb
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, resample
from imblearn.over_sampling import SMOTE
from collections import Counter
from ecg_augmentor import ECG_Augmentor 


dataset_folders = ["mit-bih", "training2017/training2017"]
records = []
labels = []
physionet_label_map = {'N': "Normal", 'A': "AFib", 'O': "Other Arrythmia"}
duration = 10 #seconds

X = []
y = []

def butter_bandpass_filter(signal, fs, low_pass=0.5, high_pass=50, order=5):
    nyq = 0.5 * fs
    low = low_pass / nyq
    high = high_pass / nyq
    b,a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def smote_resampling(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def signal_resampling(signal, target_length):
    resample(signal, target_length)

def standardise_dataset(X):
    X_scaled = np.array([(x - np.mean(x)) / np.std(x) for x in X])
    return X_scaled

def normalise_dataset(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm

def load_physionet_dataset(physionet_path, reference_file_path, X, y):

    reference_df = pd.read_csv(reference_file_path, header=None, names=['record', 'label'])

    # loading the data
    for file in os.listdir(physionet_path):
        if file.endswith(".hea"):
            record_name = file[:-4]
            records.append(record_name)

    for name in records:
        nameLabel = reference_df.loc[reference_df['record'] == name, 'label'].values[0]
        if nameLabel == "N":
            labels.append("Normal")
        elif nameLabel == "A":
            labels.append("AFib")
        else:
            labels.append("Other Arrythmia")

    # entering it in dataset
    for paths in records:
        record = wfdb.rdrecord(physionet_path + "\\" + paths)
        fs = record.fs
        samples = fs * duration
        ecg_signal = record.p_signal[:, 0]
        trunc_ecg = ecg_signal[:samples]
        # Pad if shorter than 'samples'
        if len(trunc_ecg) < samples:
            trunc_ecg = np.pad(trunc_ecg, (0, samples - len(trunc_ecg)), 'constant')
        X.append(trunc_ecg)

    for label in labels:
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y

def preprocess_dataset(physionet_path, reference_file_path, X, y, fs=300): #X and y should be empty lists

    augmentor = ECG_Augmentor(fs=fs)

    X, y = load_physionet_dataset(physionet_path, reference_file_path, X, y)
    X_filtered = np.array([butter_bandpass_filter(x, fs=fs) for x in X])

    # Augment each signal (single-signal augmentation)
    X_augmented = np.array([augmentor.augment(x) for x in X_filtered])

    #X_standardised = standardise_dataset(X_augmented)
    X_processed = normalise_dataset(X_augmented) # since this should be a dcnn, no standardisation needed
    y_processed = y

    # Apply SMOTE to balance the dataset
    X_balanced, y_balanced = smote_resampling(X_processed, y_processed)
    return X_balanced, y_balanced

X_processed, y_processed = preprocess_dataset("training2017/training2017", "training2017/training2017/REFERENCE.csv", X, y)