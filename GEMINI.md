# Project Overview

This project is a deep learning-based ECG arrhythmia classifier. It uses a convolutional neural network (CNN) built with TensorFlow and Keras to classify ECG signals into three categories: Normal, AFib, and Other.

The project is structured as follows:

-   `data_processing.py`: Handles loading and preprocessing of ECG data.
-   `data_augmentor.py`: Augments the training data to improve model robustness.
-   `data_generator.py`: Provides a custom data generator for efficiently feeding data to the model during training.
-   `model.py`: Defines, trains, and evaluates the CNN model.
-   `prediction.py`: Uses the trained model to make predictions on new ECG data.
-   `testingScripts/`: Contains scripts for testing and experimentation.

# Building and Running

## Dependencies

The project requires Python 3 and several libraries, including:

-   tensorflow
-   keras
-   numpy
-   pandas
-   scikit-learn
-   matplotlib
-   wfdb
-   neurokit2
-   focal-loss

These dependencies can be installed via pip:

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib wfdb neurokit2 focal-loss
```

## Training the Model

To train the model, run the `model.py` script:

```bash
python model.py
```

This will train the model, evaluate it on the test set, and save the trained model to `model.keras`.

## Making Predictions

To make predictions on a new ECG file, run the `prediction.py` script with the path to the file as an argument:

```bash
python prediction.py <path_to_ecg_file>
```

The script supports both CSV and WFDB file formats.

# Development Conventions

## Code Style

The project follows the PEP 8 style guide for Python code.

## Testing

The `testingScripts/` directory contains scripts for testing and experimentation. To run the tests, execute the scripts in this directory:

```bash
python testingScripts/ecg_annotations_test.py
python testingScripts/ecg_processor_test.py
```

**TODO:** Add a more robust testing framework (e.g., pytest) and write more comprehensive tests.
