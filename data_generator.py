import numpy as np
from keras.utils import Sequence

# ====================
# Custom Data Generator
# ====================
class ECGDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=64, augmentor=None, shuffle=True, signal_length=3000):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.shuffle = shuffle
        self.signal_length = signal_length
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices]

        # Apply augmentation if augmentor is provided
        if self.augmentor is not None:
            X_aug = []
            for x in X_batch:
                aug = self.augmentor.augment(x.squeeze())
                # Ensure output is always (signal_length,)
                if len(aug) > self.signal_length:
                    aug = aug[:self.signal_length]
                elif len(aug) < self.signal_length:
                    aug = np.pad(aug, (0, self.signal_length - len(aug)))
                # Normalise after augmentation
                aug = (aug - np.min(aug)) / (np.max(aug) - np.min(aug) + 1e-8)
                X_aug.append(aug)
            X_batch = np.array(X_aug, dtype=np.float32)
        else:
            # If not augmenting, ensure normalisation
            X_batch = np.array([
                (x.squeeze() - np.min(x.squeeze())) / (np.max(x.squeeze()) - np.min(x.squeeze()) + 1e-8)
                for x in X_batch
            ], dtype=np.float32)

        # Ensure correct shape for CNN
        X_batch = np.expand_dims(X_batch, axis=-1)
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
