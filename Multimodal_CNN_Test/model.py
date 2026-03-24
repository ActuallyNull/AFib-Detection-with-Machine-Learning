import tensorflow as tf
from keras import layers, models, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_processing import create_train_val_test_splits
from data_generator import ECGDataGenerator
from data_augmentor import ECG_Augmentor
from focal_loss import SparseCategoricalFocalLoss
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib

(X_train, y_train,
 X_val, y_val,
 X_test, y_test,
 f_train, f_val, f_test,
 class_names, scaler) = create_train_val_test_splits()

joblib.dump(scaler, "Multimodal_CNN_Test/scaler.pkl")
np.save("Multimodal_CNN_Test/classes.npy", np.array(class_names))

train_gen = ECGDataGenerator(X_train, y_train, f_train, augmentor=ECG_Augmentor(300, True))
val_gen   = ECGDataGenerator(X_val, y_val, f_val)
test_gen  = ECGDataGenerator(X_test, y_test, f_test)

weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

def create_model(n_features, n_classes):
    sig_in = layers.Input((3000,1))
    x = layers.Conv1D(64,15,strides=2,padding='same')(sig_in)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(3,strides=2,padding='same')(x)

    def res(x,f,s=1):
        skip = x
        x = layers.Conv1D(f,15,strides=s,padding='same',
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(f,15,padding='same',
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        if s>1:
            skip = layers.Conv1D(f,1,strides=s,padding='same')(skip)
            skip = layers.BatchNormalization()(skip)
        x = layers.Add()([skip,x])
        return layers.ReLU()(x)

    def se(x,r=16):
        f = x.shape[-1]
        s = layers.GlobalAveragePooling1D()(x)
        s = layers.Dense(f//r,activation='relu')(s)
        s = layers.Dense(f,activation='sigmoid')(s)
        s = layers.Reshape((1,f))(s)
        return layers.multiply([x,s])

    x = res(x,64); x = se(x)
    x = res(x,64); x = se(x)
    x = res(x,128,2); x = se(x)
    x = res(x,128); x = se(x)
    x = res(x,256,2); x = se(x)
    x = res(x,256)
    x = layers.GlobalAveragePooling1D()(x)

    feat_in = layers.Input((n_features,))
    y = layers.Dense(128,activation='relu')(feat_in)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(64,activation='relu')(y)
    y = layers.Dropout(0.3)(y)

    z = layers.concatenate([x,y])
    z = layers.Dense(128,activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(z)
    z = layers.Dropout(0.5)(z)
    out = layers.Dense(n_classes,activation='softmax')(z)

    return models.Model([sig_in,feat_in],out)

model = create_model(f_train.shape[1], len(class_names))
model.compile(optimizer='adam',
              loss=SparseCategoricalFocalLoss(gamma=2),
              metrics=['accuracy'])

model.fit(train_gen,
          validation_data=val_gen,
          epochs=100,
          class_weight=class_weights,
          callbacks=[
              EarlyStopping(patience=10,restore_best_weights=True),
              ReduceLROnPlateau(patience=5,factor=0.2)
          ])

y_pred = model.predict(test_gen).argmax(axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names))

model.save("Multimodal_CNN_Test/model.keras")
