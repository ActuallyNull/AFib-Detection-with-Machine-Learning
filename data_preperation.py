from data_processing import preprocess_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def create_train_val_test_splits():

    X = []
    y = []

    X_processed, y_processed = preprocess_dataset("training2017/training2017", "training2017/training2017/REFERENCE.csv", X, y)

    X_cnn = X_processed.reshape((X_processed.shape[0], X_processed.shape[1], 1))
    le = LabelEncoder()
    y_cnn = le.fit_transform(y_processed)

    X_train, X_temp, y_train, y_temp = train_test_split(X_cnn, y_cnn, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test