from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import wfdb
import scipy.signal as signal
from scipy.io import loadmat
import json
import tempfile
import os
import io
import time
from typing import Dict, List, Optional, Union
import uvicorn
from pydantic import BaseModel
import logging
from focal_loss import SparseCategoricalFocalLoss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ECG Classification API",
    description="AI-powered ECG classification for AFib and arrhythmia detection",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_loaded = False
class_names = ['AFib', 'Normal', 'Other']

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    class_probabilities: Dict[str, float]
    processing_time: float
    signal_length: int
    sampling_rate: Optional[float] = None

class PreprocessResponse(BaseModel):
    signal: List[float]
    fs: Optional[float] = None
    duration: Optional[float] = None
    original_length: int
    processed_length: int

def load_model():
    """Load the trained ECG classification model"""
    global model, model_loaded
    try:
        logger.info("Loading ECG classification model...")
        model = keras.models.load_model('../model.keras', custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss})
        model_loaded = True
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

def butter_bandpass_filter(signal_data, fs, low_pass=0.5, high_pass=50, order=5):
    """Apply bandpass filter to ECG signal"""
    try:
        nyq = 0.5 * fs
        low = low_pass / nyq
        high = high_pass / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, signal_data)
    except Exception as e:
        logger.warning(f"Filtering failed: {e}")
        return signal_data

def preprocess_ecg(signal_data, fs=300, duration=10):
    """Preprocess ECG signal for model input"""
    try:
        # Ensure signal_data is 1D array
        signal_data = np.asarray(signal_data).flatten()
        
        # Check if signal is valid
        if len(signal_data) == 0:
            raise ValueError("Empty signal data")
        
        # Resample to 300 Hz if needed
        if fs != 300:
            num_samples = int(len(signal_data) * 300 / fs)
            signal_data = signal.resample(signal_data, num_samples)
            fs = 300

        # Apply bandpass filter
        filtered_signal = butter_bandpass_filter(signal_data, fs)
        
        # Ensure filtered_signal is 1D
        filtered_signal = np.asarray(filtered_signal).flatten()
        
        # Truncate or pad to target length
        target_length = fs * duration
        if len(filtered_signal) > target_length:
            filtered_signal = filtered_signal[:target_length]
        elif len(filtered_signal) < target_length:
            filtered_signal = np.pad(filtered_signal, (0, target_length - len(filtered_signal)), 'constant')

        # Standardize the signal
        filtered_signal = filtered_signal.astype(np.float32)

        # Check for NaN or infinite values
        if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
            # Replace NaN and inf with zeros
            filtered_signal = np.nan_to_num(filtered_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape for model input
        return np.expand_dims(filtered_signal, axis=(0, -1))
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {str(e)}")

def load_wfdb_files(files: List[UploadFile]):
    """Load WFDB format files (.mat, .hea) from multiple files"""
    try:
        # Create a temporary directory to store the WFDB files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all files to the temporary directory
            saved_files = []
            for file in files:
                file_content = file.file.read()
                file.file.seek(0)  # Reset file pointer
                
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                saved_files.append(file_path)
            
            # Find the base name (without extension) for WFDB loading
            base_names = set()
            for file_path in saved_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                base_names.add(base_name)
            
            if len(base_names) != 1:
                raise ValueError("All WFDB files must have the same base name")
            
            base_name = list(base_names)[0]
            base_path = os.path.join(temp_dir, base_name)
            
            # Load the WFDB record
            record = wfdb.rdrecord(base_path)
            
            # Get the first channel (lead)
            signal_data = record.p_signal[:, 0]
            fs = record.fs
            
            return signal_data, fs
            
    except Exception as e:
        logger.error(f"Error loading WFDB files: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading WFDB files: {str(e)}")

def load_wfdb_file(file_path: str, file_name: str):
    """Load WFDB format files (.mat, .hea)"""
    try:
        # Remove extension for WFDB loading
        base_name = os.path.splitext(file_name)[0]
        base_path = os.path.join(file_path, base_name)
        
        # Load the WFDB record
        record = wfdb.rdrecord(base_path)
        
        # Get the first channel (lead)
        signal_data = record.p_signal[:, 0]
        fs = record.fs
        
        return signal_data, fs
        
    except Exception as e:
        logger.error(f"Error loading WFDB file: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading WFDB file: {str(e)}")

def load_csv_file(file_content: bytes):
    """Load CSV format files"""
    try:
        # Try to read CSV content
        content = file_content.decode('utf-8')
        
        # Handle different CSV formats
        if ',' in content:
            df = pd.read_csv(io.StringIO(content))
        else:
            # Assume space or tab separated
            df = pd.read_csv(io.StringIO(content), sep=None, engine='python')
        
        # Get the first column as signal
        signal_data = df.iloc[:, 0].values
        
        # Assume default sampling rate for CSV
        fs = 300
        
        return signal_data, fs
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading CSV file: {str(e)}")

def load_txt_file(file_content: bytes):
    """Load TXT format files"""
    try:
        # Try to read text content
        content = file_content.decode('utf-8')
        
        # Split by lines and convert to numbers
        lines = content.strip().split('\n')
        signal_data = []
        
        for line in lines:
            try:
                # Handle different separators
                if ',' in line:
                    values = line.split(',')
                elif '\t' in line:
                    values = line.split('\t')
                else:
                    values = line.split()
                
                # Take the first value from each line
                if values:
                    signal_data.append(float(values[0]))
            except ValueError:
                continue
        
        if not signal_data:
            raise ValueError("No valid numeric data found")
        
        signal_data = np.array(signal_data)
        
        # Assume default sampling rate for TXT
        fs = 300
        
        return signal_data, fs
        
    except Exception as e:
        logger.error(f"Error loading TXT file: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading TXT file: {str(e)}")

def load_ecg_file(file: UploadFile):
    """Load ECG file based on its format"""
    try:
        file_content = file.file.read()
        file.file.seek(0)  # Reset file pointer
        
        file_extension = file.filename.lower().split('.')[-1]
        
        if file_extension in ['mat', 'hea', 'dat']:
            # Save to temporary file for WFDB loading
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                signal_data, fs = load_wfdb_file(os.path.dirname(tmp_path), os.path.basename(tmp_path))
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        elif file_extension == 'csv':
            signal_data, fs = load_csv_file(file_content)
            
        elif file_extension == 'txt':
            signal_data, fs = load_txt_file(file_content)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
        
        return signal_data, fs
        
    except Exception as e:
        logger.error(f"Error loading ECG file: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading ECG file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "class_names": class_names,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": model.count_params()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_ecg(file: UploadFile = File(...)):
    """Predict ECG classification"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Load ECG file
        signal_data, fs = load_ecg_file(file)
        
        # Preprocess for model
        processed_signal = preprocess_ecg(signal_data, fs)
        
        # Make prediction
        predictions = model.predict(processed_signal, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_label = class_names[predicted_class_idx]
        
        # Create class probabilities dictionary
        class_probabilities = {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            label=predicted_label,
            confidence=confidence,
            class_probabilities=class_probabilities,
            processing_time=processing_time,
            signal_length=len(signal_data),
            sampling_rate=fs
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-wfdb", response_model=PredictionResponse)
async def predict_wfdb_ecg(files: List[UploadFile] = File(...)):
    """Predict ECG classification for WFDB files"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="WFDB requires at least .hea and .mat files")
    
    start_time = time.time()
    
    try:
        # Load WFDB files
        signal_data, fs = load_wfdb_files(files)
        
        # Preprocess for model
        processed_signal = preprocess_ecg(signal_data, fs)
        
        # Make prediction
        predictions = model.predict(processed_signal, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_label = class_names[predicted_class_idx]
        
        # Create class probabilities dictionary
        class_probabilities = {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            label=predicted_label,
            confidence=confidence,
            class_probabilities=class_probabilities,
            processing_time=processing_time,
            signal_length=len(signal_data),
            sampling_rate=fs
        )
        
    except Exception as e:
        logger.error(f"WFDB prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"WFDB prediction failed: {str(e)}")

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_ecg_for_visualization(file: UploadFile = File(...)):
    """Preprocess ECG for visualization (returns cleaned signal)"""
    try:
        # Load ECG file
        signal_data, fs = load_ecg_file(file)
        
        # Apply preprocessing but keep original length for visualization
        original_length = len(signal_data)
        
        # Apply bandpass filter
        filtered_signal = butter_bandpass_filter(signal_data, fs)
        
        # Normalize for visualization
        signal_min = np.min(filtered_signal)
        signal_max = np.max(filtered_signal)
        signal_range = signal_max - signal_min + 1e-8
        
        if signal_range <= 0:
            normalized_signal = np.zeros_like(filtered_signal)
        else:
            normalized_signal = (filtered_signal - signal_min) / signal_range
        
        # Calculate duration
        duration = original_length / fs if fs else None
        
        return PreprocessResponse(
            signal=normalized_signal.tolist(),
            fs=fs,
            duration=duration,
            original_length=original_length,
            processed_length=len(normalized_signal)
        )
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.post("/preprocess-wfdb", response_model=PreprocessResponse)
async def preprocess_wfdb_for_visualization(files: List[UploadFile] = File(...)):
    """Preprocess WFDB files for visualization (returns cleaned signal)"""
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="WFDB requires at least .hea and .mat files")
        
        # Load WFDB files
        signal_data, fs = load_wfdb_files(files)
        
        # Apply preprocessing but keep original length for visualization
        original_length = len(signal_data)
        
        # Apply bandpass filter
        filtered_signal = butter_bandpass_filter(signal_data, fs)
        
        # Normalize for visualization
        signal_min = np.min(filtered_signal)
        signal_max = np.max(filtered_signal)
        signal_range = signal_max - signal_min + 1e-8
        
        if signal_range <= 0:
            normalized_signal = np.zeros_like(filtered_signal)
        else:
            normalized_signal = (filtered_signal - signal_min) / signal_range
        
        # Calculate duration
        duration = original_length / fs if fs else None
        
        return PreprocessResponse(
            signal=normalized_signal.tolist(),
            fs=fs,
            duration=duration,
            original_length=original_length,
            processed_length=len(normalized_signal)
        )
        
    except Exception as e:
        logger.error(f"WFDB preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"WFDB preprocessing failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict_ecg(files: List[UploadFile] = File(...)):
    """Batch predict multiple ECG files"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            # Load ECG file
            signal_data, fs = load_ecg_file(file)
            
            # Preprocess for model
            processed_signal = preprocess_ecg(signal_data, fs)
            
            # Make prediction
            predictions = model.predict(processed_signal, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_label = class_names[predicted_class_idx]
            
            # Create class probabilities dictionary
            class_probabilities = {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
            
            results.append({
                "filename": file.filename,
                "label": predicted_label,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "signal_length": len(signal_data),
                "sampling_rate": fs,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}

SHOWCASE_ECG_DIR = "../showcase_ecgs"

@app.post("/admin/upload-showcase-ecg")
async def upload_showcase_ecg(files: List[UploadFile] = File(...)):
    """Upload showcase ECG files"""
    if not os.path.exists(SHOWCASE_ECG_DIR):
        os.makedirs(SHOWCASE_ECG_DIR)
    
    saved_files = []
    for file in files:
        file_path = os.path.join(SHOWCASE_ECG_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        saved_files.append(file.filename)
    
    return JSONResponse(content={"message": "Files uploaded successfully", "filenames": saved_files})

@app.get("/showcase-ecgs")
async def get_showcase_ecgs():
    """Get all showcase ECGs"""
    if not os.path.exists(SHOWCASE_ECG_DIR):
        return JSONResponse(content={"error": "Showcase directory not found"}, status_code=404)
    
    ecg_data = []
    files = os.listdir(SHOWCASE_ECG_DIR)
    
    # Group files by base name
    file_groups = {}
    for f in files:
        base_name = os.path.splitext(f)[0]
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(f)
        
    for base_name, file_list in file_groups.items():
        if len(file_list) >= 2: # .hea and .mat/.dat
            try:
                record = wfdb.rdrecord(os.path.join(SHOWCASE_ECG_DIR, base_name))
                signal_data = record.p_signal[:, 0].tolist()
                fs = record.fs
                
                # Get file stats
                mat_file = next((f for f in file_list if f.endswith(('.mat', '.dat'))), None)
                if mat_file:
                    file_path = os.path.join(SHOWCASE_ECG_DIR, mat_file)
                    file_size = os.path.getsize(file_path)
                    last_modified = os.path.getmtime(file_path)
                else:
                    file_size = -1
                    last_modified = -1

                ecg_data.append({
                    "filename": base_name,
                    "signal": signal_data,
                    "fs": fs,
                    "file_size": file_size,
                    "last_modified": last_modified,
                    "file_type": "WFDB"
                })
            except Exception as e:
                logger.error(f"Error processing showcase file {base_name}: {e}")

    return JSONResponse(content=ecg_data)

@app.delete("/admin/delete-showcase-ecg/{filename}")
async def delete_showcase_ecg(filename: str):
    """Delete a showcase ECG file"""
    if not os.path.exists(SHOWCASE_ECG_DIR):
        raise HTTPException(status_code=404, detail="Showcase directory not found")

    base_name = os.path.splitext(filename)[0]
    files_to_delete = [f for f in os.listdir(SHOWCASE_ECG_DIR) if os.path.splitext(f)[0] == base_name]

    if not files_to_delete:
        raise HTTPException(status_code=404, detail=f"Files for {filename} not found")

    for f in files_to_delete:
        os.remove(os.path.join(SHOWCASE_ECG_DIR, f))

    return JSONResponse(content={"message": f"Files for {filename} deleted successfully"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
