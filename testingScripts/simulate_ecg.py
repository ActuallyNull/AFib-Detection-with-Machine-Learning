import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
import os
from scipy.io import savemat

def simulate_ecg(num_cycles=5, fs=300, hr=60, rhythm="normal",
                 output_prefix="synthetic_ecg", output_dir="testingScripts"):
    """
    Simulate synthetic ECG with NeuroKit2 and save to WFDB (.hea + .mat), CSV, and TXT.

    Parameters
    ----------
    num_cycles : int
        Number of cardiac cycles to simulate.
    fs : int
        Sampling frequency in Hz.
    hr : int
        Heart rate in beats per minute.
    rhythm : str
        Rhythm type: "normal", "afib", or "svt".
    output_prefix : str
        Base name of output files.
    output_dir : str
        Directory to save files.
    """

    # duration in seconds
    duration = num_cycles * (60 / hr)
    n_samples = int(round(duration * fs))
    sim_length = n_samples + fs  # pad to ensure enough samples

    # Map rhythm types to NK2 methods
    rhythm_map = {
        "normal": "ecgsyn",
        "afib": "fibrillation",
        "svt": "tachycardia"
    }
    method = rhythm_map.get(rhythm.lower(), "ecgsyn")

    # Simulate ECG
    ecg = nk.ecg_simulate(length=sim_length, sampling_rate=fs,
                          heart_rate=hr, method=method)
    ecg = ecg[:n_samples]  # trim to exact length

    # Prepare output filenames
    record_name = output_prefix.replace(".", "_")
    os.makedirs(output_dir, exist_ok=True)

    # Save WFDB record (.hea)
    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["mV"],
        sig_name=["II"],   # Lead II only
        p_signal=ecg.reshape(-1, 1),
        write_dir=output_dir,
        fmt=["16"],
    )

    # Save .mat file
    mat_file_name = os.path.join(output_dir, f"{record_name}.mat")
    savemat(mat_file_name, {'val': ecg.reshape(-1, 1)})

    # Remove the .dat file
    dat_file_path = os.path.join(output_dir, f"{record_name}.dat")
    if os.path.exists(dat_file_path):
        os.remove(dat_file_path)


    # Save CSV
    pd.DataFrame({"ECG": ecg}).to_csv(
        os.path.join(output_dir, f"{record_name}.csv"), index=False)

    # Save TXT
    np.savetxt(os.path.join(output_dir, f"{record_name}.txt"), ecg)

    print(f"[+] Saved {record_name}.hea, .mat, .csv, and .txt in {output_dir}")
    print(f"[i] Rhythm: {rhythm}, HR: {hr} bpm, Samples: {n_samples}")

if __name__ == "__main__":
    # Example usage
    simulate_ecg(num_cycles=5, fs=300, hr=60,
                 rhythm="normal", output_prefix="synthetic_normal", output_dir="testingScripts")
    simulate_ecg(num_cycles=5, fs=300, hr=90,
                 rhythm="afib", output_prefix="synthetic_afib", output_dir="testingScripts")
    simulate_ecg(num_cycles=5, fs=300, hr=150,
                 rhythm="svt", output_prefix="synthetic_svt", output_dir="testingScripts")