import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_processing import create_train_val_test_splits
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from keras import regularizers
from focal_loss import SparseCategoricalFocalLoss
import numpy as np
from data_generator import ECGDataGenerator
from data_augmentor import ECG_Augmentor

X_train, y_train, X_val, y_val, X_test, y_test, le = create_train_val_test_splits()

train_augmentor = ECG_Augmentor(fs=300, strong=True)
train_gen = ECGDataGenerator(X_train, y_train, batch_size=64, augmentor=train_augmentor, shuffle=True)
val_gen = ECGDataGenerator(X_val, y_val, batch_size=64, augmentor=None, shuffle=False)

unique, counts = np.unique(y_train, return_counts=True)
print("Class distribution in y_train:", dict(zip(unique, counts)))

weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
class_weights = dict(enumerate(weights))
class_weights[2] *= 1.3 # give more weight to other arrhythmia since it gets misclassfied a lot more

callback = [
    EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
]

def conv_block(x, filters, kernel_size, pool_size=2, dropout=0.2):
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    x = layers.Dropout(dropout)(x)
    return x

def residual_block(x, filters, kernel_size=15, strides=1):
    x_skip = x
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    if strides > 1:
        x_skip = layers.Conv1D(filters, 1, strides=strides, padding='same')(x_skip)
        x_skip = layers.BatchNormalization()(x_skip)
    x = layers.Add()([x_skip, x])
    x = layers.ReLU()(x)
    return x

def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, filters))(se)
    return layers.multiply([input_tensor, se])

def ecg_cnn(input_length, num_classes=3): # 0: afib, 1: normal, 2: other arrhythmia
    inputs = layers.Input(shape=(input_length, 1))

    x = layers.Conv1D(64, 15, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = residual_block(x, 64)
    x = squeeze_excite_block(x)
    x = residual_block(x, 64)
    x = squeeze_excite_block(x)
    x = residual_block(x, 128, strides=2)
    x = squeeze_excite_block(x)
    x = residual_block(x, 128)
    x = squeeze_excite_block(x)
    x = residual_block(x, 256, strides=2)
    x = squeeze_excite_block(x)
    x = residual_block(x, 256)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

test_gen = ECGDataGenerator(X_test, y_test, batch_size=64, augmentor=None, shuffle=False)

n_models = 3
ensemble_preds = np.zeros((len(y_test), 3))  # 3 = num_classes
ensemble_test_acc = 0

for seed in range(n_models):
    print(f"\nTraining ensemble model {seed+1}/{n_models}")
    tf.keras.utils.set_random_seed(seed)
    model = ecg_cnn(input_length=3000)
    model.compile(
        optimizer='adam',
        loss=SparseCategoricalFocalLoss(gamma=2),
        metrics=['accuracy']
    )
    model.fit(
        train_gen,
        epochs=100,
        callbacks=callback,
        class_weight=class_weights,
        validation_data=val_gen,
        verbose=0  # Suppress output for brevity
    )

    test_loss, test_acc = model.evaluate(test_gen)
    ensemble_test_acc += test_acc
    # Predict on test set
    y_pred = model.predict(test_gen)
    ensemble_preds += y_pred

ensemble_preds /= n_models
ensemble_pred_classes = ensemble_preds.argmax(axis=1)

ensemble_test_acc /= n_models
print(f"Average test accuracy: {ensemble_test_acc:.4f}")

print("\nEnsemble Confusion Matrix:")
print(confusion_matrix(y_test, ensemble_pred_classes))
print("Ensemble Classification Report:")
print(classification_report(y_test, ensemble_pred_classes))

# ROC and PR curves for ensemble
y_test_binary = label_binarize(y_test, classes=[0,1,2])

for i in range(ensemble_preds.shape[1]):
    RocCurveDisplay.from_predictions(y_test_binary[:,i], ensemble_preds[:,i], name=f"Class {i}")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("Ensemble OvR ROC Curve")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
for i in range(ensemble_preds.shape[1]):
    precision, recall, _ = precision_recall_curve(y_test_binary[:, i], ensemble_preds[:, i])
    auc_score = auc(recall, precision)
    plt.plot(recall, precision, label=f'Class {i} (AUC = {auc_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Ensemble Precision-Recall Curve (OvR)')
plt.legend()
plt.show()