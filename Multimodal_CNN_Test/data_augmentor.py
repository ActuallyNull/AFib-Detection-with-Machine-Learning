import numpy as np
from scipy.signal import resample

class ECG_Augmentor:
    def __init__(self, fs=360, strong=False):
        self.fs = fs
        self.strong = strong

    def add_gaussian_noise(self, signal, noise_level=0.01):
        max_val = np.max(np.abs(signal))
        scale = max(noise_level * max_val, 1e-6)
        noise = np.random.normal(0, scale, size=signal.shape)
        return signal + noise

    def baseline_wander(self, signal, freq=0.3, amp=0.05):
        t = np.arange(len(signal)) / self.fs
        drift = amp * np.sin(2 * np.pi * freq * t)
        return signal + drift

    def time_shift(self, signal, shift_max=0.05):
        shift = int(np.random.uniform(-shift_max, shift_max) * len(signal))
        return np.roll(signal, shift)

    def time_scale(self, signal, scale_range=(0.95, 1.05)):
        scale = np.random.uniform(*scale_range)
        new_len = int(len(signal) * scale)
        scaled = resample(signal, new_len)
        if len(scaled) > len(signal):
            return scaled[:len(signal)]
        return np.pad(scaled, (0, len(signal) - len(scaled)))

    def amplitude_scale(self, signal, scale_range=(0.95, 1.05)):
        return signal * np.random.uniform(*scale_range)

    def amplitude_shift(self, signal, shift_range=(-0.05, 0.05)):
        shift = np.random.uniform(*shift_range) * np.max(signal)
        return signal + shift

    def random_crop(self, signal, crop_size=0.95):
        orig_len = len(signal)
        crop_len = int(orig_len * crop_size)
        start = np.random.randint(0, orig_len - crop_len)
        cropped = signal[start:start+crop_len]
        return np.pad(cropped, (0, orig_len - crop_len))

    def invert(self, signal):
        return -signal

    def mixup(self, sig1, sig2, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        return lam * sig1 + (1 - lam) * sig2

    def augment(self, signal, mix_signal=None):
        aug = signal.copy()

        if np.random.rand() < 0.5:
            aug = self.add_gaussian_noise(aug)
        if np.random.rand() < 0.2:
            aug = self.baseline_wander(aug)
        if np.random.rand() < 0.1:
            aug = self.time_shift(aug)
        if np.random.rand() < 0.1:
            aug = self.time_scale(aug)
        if np.random.rand() < 0.2:
            aug = self.amplitude_scale(aug)
        if np.random.rand() < 0.1:
            aug = self.amplitude_shift(aug)

        if self.strong:
            if np.random.rand() < 0.1:
                aug = self.random_crop(aug)
            if np.random.rand() < 0.05:
                aug = self.invert(aug)

        if mix_signal is not None and np.random.rand() < 0.1:
            aug = self.mixup(aug, mix_signal)

        return aug
