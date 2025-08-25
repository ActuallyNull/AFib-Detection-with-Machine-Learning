import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os

def plot_wfdb(record_name):
    """
    Load a .mat file and plot the ECG signal.
    """
    # Construct the path to the .mat file
    mat_file_path = f"{record_name}.mat"

    # Load the .mat file
    mat_data = loadmat(mat_file_path)
    signals = mat_data['val']

    # Basic info for plotting
    print(f"Record info: Loaded from {mat_file_path}")

    # Plot the first channel (Lead II)
    plt.figure(figsize=(12, 4))
    plt.plot(signals, label="Lead II")
    plt.title(f"Simulated ECG - {os.path.basename(record_name)}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # This script now assumes that the .mat files are in the 'testingScripts' directory
    plot_wfdb("testingScripts/synthetic_normal")
    plot_wfdb("testingScripts/synthetic_afib")
    plot_wfdb("testingScripts/synthetic_svt")