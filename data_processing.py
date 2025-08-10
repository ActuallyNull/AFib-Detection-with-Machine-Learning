from sklearn.model_selection import train_test_split
import wfdb
import numpy as np
import os
import pandas as pd

dataset_folders = ["mit-bih", "training2017/training2017"]
records = []
labels = []
physionet_label_map = {'N': "Normal", 'A': "AFib", 'O': "Other Arrythmia"}
duration = 10 #seconds

X = []
y = []

def standardise_dataset(X):
    X_scaled = np.array([(x - np.mean(x)) / np.std(x) for x in X])
    return X_scaled

from sklearn.preprocessing import MinMaxScaler

def normalise_dataset(X):
    # Reshape X to 2D if needed (n_samples, n_features)
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

def process_physionet_dataset(physionet_path, reference_file_path, X, y): #X and y should be empty lists

    X, y = load_physionet_dataset(physionet_path, reference_file_path, X, y)
    X = standardise_dataset(X)
    X = normalise_dataset(X)
    return X, y

X, y = process_physionet_dataset("training2017/training2017", "training2017/training2017/REFERENCE.csv", X, y)

