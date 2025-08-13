import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from data_processing import create_train_val_test_splits
from data_augmentor import ECG_Augmentor

# ====================
# Custom Data Generator
# ====================
class ECGDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=64, augmentor=None, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.shuffle = shuffle
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
            X_batch = np.array([self.augmentor.augment(x.squeeze()) for x in X_batch])
        
        # Ensure correct shape for CNN
        X_batch = np.expand_dims(X_batch, axis=-1)
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
