from sklearn.model_selection import train_test_split
import wfdb
import numpy as np
import os

dataset_folders = ["mit-bih", "training2017/training2017"]
records = []

for folder in dataset_folders:
    for file in os.listdir(folder):
        if file.endswith(".hea"): # .hea is a common extension in all dataset_folders
            record_name = file[:-4] #splicing (removing) extension name
            records.append(record_name)

X = []

for paths in records:
    record = wfdb.rdrecord(paths)
    ecg_signal = record.p_signal[:, 0]
    X.append(ecg_signal)

