from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import wfdb
import scipy.signal as sp_signal
from scipy.io import loadmat
import json
import tempfile
import os
import io
import time
from typing import Dict, List, Optional
import uvicorn
from pydantic import BaseModel
import logging
from focal_loss import SparseCategoricalFocalLoss
import re

# -----------------------------
# Config & Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="ECG Classification API",
    description="AI-powered ECG classification for AFib and arrhythmia detection",
    version="2.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Globals
# -----------------------------
model = None
model_loaded = False
# Match training label encoder order (alphabetical): ['AFib', 'Normal', 'Other Arrythmia']
class_names = ['AFib', 'Normal', 'Other Arrythmia']

TARGET_FS = 300
TARGET_DURATION_SEC = 10
TARGET_SAMPLES = TARGET_FS * TARGET_DURATION_SEC

SHOWCASE_ECG_BASE_DIR = "showcase_ecgs"
PREDICTION_SHOWCASE_DIR = os.path.join(SHOWCASE_ECG_BASE_DIR, "prediction_showcase")
VIEWER_SHOWCASE_DIR = os.path.join(SHOWCASE_ECG_BASE_DIR, "viewer_showcase")

# -----------------------------
# Models / Schemas
# -----------------------------
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

class PredictionRequest(BaseModel):
    filename: str

class RenameRequest(BaseModel):
    new_filename: str

# -----------------------------
# Utilities
# -----------------------------

def sanitize_wfdb_name(name: str) -> str:
    # Only allow letters, numbers, and underscores
    return re.sub(r'[^A-Za-z0-9_]', '_', name)


def load_model():
    global model, model_loaded
    try:
        logger.info("Loading ECG classification model...")
        model = keras.models.load_model(
            'model.keras',
            custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss}
                )
        model_loaded = True
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return False

@app.on_event("startup")
async def startup_event():
    load_model()

def butter_bandpass_filter(signal_data, fs, low_pass=0.5, high_pass=50, order=5):
    """Apply the same bandpass filter used in training (0.5–50 Hz)."""
    try:
        nyq = 0.5 * fs
        low = low_pass / nyq
        high = high_pass / nyq
        b, a = sp_signal.butter(order, [low, high], btype='band')
        return sp_signal.filtfilt(b, a, signal_data)
    except Exception as e:
        logger.warning(f"Filtering failed, passing through raw signal. Error: {e}")
        return signal_data

def _resample_to_target_fs(x: np.ndarray, src_fs: float, dst_fs: int = TARGET_FS) -> np.ndarray:
    """Resample 1D signal to dst_fs using polyphase (better than FFT resample for ECG)."""
    if src_fs == dst_fs:
        return x
    # use rational approximation for up/down factors
    gcd = np.gcd(int(src_fs), int(dst_fs))
    up = dst_fs // gcd
    down = int(src_fs) // gcd
    return sp_signal.resample_poly(x, up, down)

def _truncate_or_pad(x: np.ndarray, length: int = TARGET_SAMPLES) -> np.ndarray:
    if len(x) > length:
        return x[:length]
    if len(x) < length:
        return np.pad(x, (0, length - len(x)), mode='constant')
    return x

def preprocess_ecg(signal_data: np.ndarray, fs: float) -> np.ndarray:
    """Match training pipeline exactly:
       - Resample to 300 Hz if needed
       - Bandpass 0.5–50 Hz
       - Truncate/pad to 10 s (3000 samples)
       - No normalisation/standardisation
       - Return shape (1, 3000, 1)
    """
    try:
        x = np.asarray(signal_data, dtype=np.float32).flatten()
        if x.size == 0:
            raise ValueError("Empty signal data")

        # Resample if needed
        if fs != TARGET_FS:
            x = _resample_to_target_fs(x, fs, TARGET_FS)
            fs = TARGET_FS

        # Bandpass like training
        x = butter_bandpass_filter(x, fs)

        # Truncate/pad to training window
        x = _truncate_or_pad(x, TARGET_SAMPLES)

        # Clean NaNs/Infs (rare but safe)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # (batch, time, channels)
        x = np.expand_dims(x, axis=(0, -1)).astype(np.float32)
        return x
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {str(e)}")

# -----------------------------
# Robust WFDB loaders
# -----------------------------
def _wfdb_load_base(base_path_no_ext: str):
    """Load waveform and fs from a WFDB base path (no extension). Prefers .mat files first."""
    try:
        # Try rdrecord first (can read .mat)
        rec = wfdb.rdrecord(base_path_no_ext)
        fs = float(rec.fs)
        sig = rec.p_signal
        sig1 = sig[:, 0] if sig.ndim == 2 else sig.squeeze()
        return sig1.astype(np.float32), fs
    except Exception as e1:
        # Fallback to rdsamp (classic .dat)
        try:
            sig, fields = wfdb.rdsamp(base_path_no_ext)
            fs = float(fields.get('fs')) if isinstance(fields, dict) else float(fields.fs)
            sig1 = sig[:, 0] if sig.ndim == 2 else sig.squeeze()
            return sig1.astype(np.float32), fs
        except Exception as e2:
            raise RuntimeError(f"WFDB read failed (rdrecord: {e1}; rdsamp: {e2})")

def _infer_base_path(temp_dir: str, any_filename: str) -> str:
    """Given a directory and any filename ('name.ext'), return base path 'dir/name'."""
    base = os.path.splitext(any_filename)[0]
    return os.path.join(temp_dir, base)

def _ensure_pair_exists(dir_path: str, base_name: str):
    """Ensure both header and signal file exist for WFDB MAT/HEA or DAT/HEA."""
    files = os.listdir(dir_path)
    stems = {os.path.splitext(f)[0] for f in files}
    if base_name not in stems:
        raise FileNotFoundError(f"No files found for base '{base_name}' in {dir_path}")

    # Accept either .mat or .dat as signal file (plus .hea)
    has_header = any(f == f"{base_name}.hea" for f in files)
    has_mat = any(f == f"{base_name}.mat" for f in files)
    has_dat = any(f == f"{base_name}.dat" for f in files)
    if not has_header or not (has_mat or has_dat):
        raise FileNotFoundError(f"Missing required WFDB components for '{base_name}' (need .hea + .mat/.dat)")

def load_wfdb_files(files: List[UploadFile]):
    """Load WFDB from uploaded files (expect .hea + .mat/.dat)."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            saved = []
            for file in files:
                content = file.file.read()
                file.file.seek(0)
                path = os.path.join(temp_dir, file.filename)
                with open(path, 'wb') as f:
                    f.write(content)
                saved.append(path)

            base_names = {os.path.splitext(os.path.basename(p))[0] for p in saved}
            if len(base_names) != 1:
                raise ValueError("All WFDB files must share the same base name")
            base = list(base_names)[0]

            _ensure_pair_exists(temp_dir, base)
            base_path = os.path.join(temp_dir, base)
            signal_data, fs = _wfdb_load_base(base_path)
            return signal_data, fs
    except Exception as e:
        logger.error(f"Error loading WFDB files: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading WFDB files: {str(e)}")

def load_wfdb_file(dir_path: str, base_name_no_ext: str):
    """Load WFDB by directory & base name (no extension)."""
    try:
        _ensure_pair_exists(dir_path, base_name_no_ext)
        base_path = os.path.join(dir_path, base_name_no_ext)
        signal_data, fs = _wfdb_load_base(base_path)
        return signal_data, fs
    except Exception as e:
        logger.error(f"Error loading WFDB file: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading WFDB file: {str(e)}")

# -----------------------------
# CSV/TXT loaders (non-WFDB)
# -----------------------------
def load_csv_file(file_content: bytes):
    try:
        text = file_content.decode('utf-8', errors='ignore')
        # Try CSV, else autodetect sep
        if ',' in text:
            df = pd.read_csv(io.StringIO(text))
        else:
            df = pd.read_csv(io.StringIO(text), sep=None, engine='python')
        signal_data = df.iloc[:, 0].astype(float).values
        fs = TARGET_FS  # assume 300 if not provided
        return signal_data, fs
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading CSV file: {str(e)}")

def load_txt_file(file_content: bytes):
    try:
        text = file_content.decode('utf-8', errors='ignore')
        vals = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                first = line.split(',')[0]
            elif '\t' in line:
                first = line.split('\t')[0]
            else:
                first = line.split()[0]
            try:
                vals.append(float(first))
            except:
                continue
        if not vals:
            raise ValueError("No valid numeric data found")
        signal_data = np.array(vals, dtype=np.float32)
        fs = TARGET_FS
        return signal_data, fs
    except Exception as e:
        logger.error(f"Error loading TXT file: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading TXT file: {str(e)}")

def load_ecg_file(file: UploadFile):
    """Generic loader for standalone files. For WFDB, the recommended path is uploading both files via /predict-wfdb."""
    try:
        content = file.file.read()
        file.file.seek(0)
        ext = file.filename.lower().split('.')[-1]

        if ext in ['csv']:
            return load_csv_file(content)
        if ext in ['txt']:
            return load_txt_file(content)

        if ext in ['hea', 'mat', 'dat']:
            # User provided only one piece of WFDB; we need the pair.
            # Save and attempt to read only if the counterpart exists in same folder (rare here).
            with tempfile.TemporaryDirectory() as temp_dir:
                one_path = os.path.join(temp_dir, file.filename)
                with open(one_path, 'wb') as f:
                    f.write(content)
                base = os.path.splitext(file.filename)[0]
                # If user uploaded just one file, we cannot proceed reliably.
                raise HTTPException(status_code=400, detail="For WFDB please upload both .hea and .mat/.dat via /predict-wfdb")
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading ECG file: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading ECG file: {str(e)}")

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_loaded, "timestamp": time.time()}

@app.get("/model-info")
async def model_info():
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
async def predict_ecg(request: PredictionRequest):
    """Predict from a WFDB pair stored in the prediction showcase folder."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    try:
        # request.filename should be the base name without extension
        base_name = os.path.splitext(request.filename)[0]
        signal_data, fs = load_wfdb_file(PREDICTION_SHOWCASE_DIR, base_name)

        x = preprocess_ecg(signal_data, fs)
        preds = model.predict(x, verbose=0)[0]
        cls_idx = int(np.argmax(preds))
        response = PredictionResponse(
            label=class_names[cls_idx],
            confidence=float(preds[cls_idx]),
            class_probabilities={class_names[i]: float(preds[i]) for i in range(len(class_names))},
            processing_time=time.time() - start,
            signal_length=len(signal_data),
            sampling_rate=fs
        )
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-wfdb", response_model=PredictionResponse)
async def predict_wfdb_ecg(files: List[UploadFile] = File(...)):
    """Predict from uploaded WFDB pair (.hea + .mat/.dat)."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="WFDB requires at least .hea and .mat/.dat")
    start = time.time()
    try:
        signal_data, fs = load_wfdb_files(files)
        x = preprocess_ecg(signal_data, fs)
        preds = model.predict(x, verbose=0)[0]
        cls_idx = int(np.argmax(preds))
        return PredictionResponse(
            label=class_names[cls_idx],
            confidence=float(preds[cls_idx]),
            class_probabilities={class_names[i]: float(preds[i]) for i in range(len(class_names))},
            processing_time=time.time() - start,
            signal_length=len(signal_data),
            sampling_rate=fs
        )
    except Exception as e:
        logger.error(f"WFDB prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"WFDB prediction failed: {str(e)}")

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_ecg_for_visualization(file: UploadFile = File(...)):
    """Return a filtered & min-max scaled version for plotting ONLY (does not affect model)."""
    try:
        signal_data, fs = load_ecg_file(file)
        if signal_data is None:
            raise HTTPException(status_code=400, detail="Unable to load ECG data from provided file")

        orig_len = len(signal_data)
        # For viewer: keep original sampling, just filter then min-max to [0,1].
        filtered = butter_bandpass_filter(np.asarray(signal_data, dtype=np.float32), fs)
        smin, smax = float(np.min(filtered)), float(np.max(filtered))
        rng = (smax - smin) if (smax > smin) else 1.0
        normalized = (filtered - smin) / rng
        duration = (orig_len / fs) if fs else None

        return PreprocessResponse(
            signal=normalized.astype(np.float32).tolist(),
            fs=fs,
            duration=duration,
            original_length=orig_len,
            processed_length=len(normalized)
        )
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.post("/preprocess-wfdb", response_model=PreprocessResponse)
async def preprocess_wfdb_for_visualization(files: List[UploadFile] = File(...)):
    """Viewer helper for WFDB pair: filter + min-max for plotting (no model)."""
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="WFDB requires at least .hea and .mat/.dat")
        signal_data, fs = load_wfdb_files(files)
        orig_len = len(signal_data)
        filtered = butter_bandpass_filter(np.asarray(signal_data, dtype=np.float32), fs)
        smin, smax = float(np.min(filtered)), float(np.max(filtered))
        rng = (smax - smin) if (smax > smin) else 1.0
        normalized = (filtered - smin) / rng
        duration = (orig_len / fs) if fs else None

        return PreprocessResponse(
            signal=normalized.astype(np.float32).tolist(),
            fs=fs,
            duration=duration,
            original_length=orig_len,
            processed_length=len(normalized)
        )
    except Exception as e:
        logger.error(f"WFDB preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"WFDB preprocessing failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict_ecg(files: List[UploadFile] = File(...)):
    """Batch predict for non-WFDB files (CSV/TXT). For WFDB, use /predict-wfdb."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for file in files:
        try:
            signal_data, fs = load_ecg_file(file)
            x = preprocess_ecg(signal_data, fs)
            preds = model.predict(x, verbose=0)[0]
            cls_idx = int(np.argmax(preds))
            results.append({
                "filename": file.filename,
                "label": class_names[cls_idx],
                "confidence": float(preds[cls_idx]),
                "class_probabilities": {class_names[i]: float(preds[i]) for i in range(len(class_names))},
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

# -----------------------------
# Showcase admin helpers
# -----------------------------
async def upload_ecg_to_folder(files: List[UploadFile], target_folder: str):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    saved_files = []
    for file in files:
        file_path = os.path.join(target_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        saved_files.append(file.filename)
    return JSONResponse(content={"message": "Files uploaded successfully", "filenames": saved_files})

@app.post("/admin/upload-prediction-ecg")
async def upload_prediction_ecg(files: List[UploadFile] = File(...)):
    return await upload_ecg_to_folder(files, PREDICTION_SHOWCASE_DIR)

@app.post("/admin/upload-viewer-ecg")
async def upload_viewer_ecg(files: List[UploadFile] = File(...)):
    return await upload_ecg_to_folder(files, VIEWER_SHOWCASE_DIR)

@app.get("/showcase-ecgs")
async def get_showcase_ecgs(folder_type: str = "prediction"):
    if folder_type == "prediction":
        target_dir = PREDICTION_SHOWCASE_DIR
    elif folder_type == "viewer":
        target_dir = VIEWER_SHOWCASE_DIR
    else:
        raise HTTPException(status_code=400, detail="Invalid folder_type. Must be 'prediction' or 'viewer'.")

    if not os.path.exists(target_dir):
        return JSONResponse(content=[], status_code=200)

    ecg_data = []
    files = os.listdir(target_dir)
    groups = {}
    for f in files:
        base = os.path.splitext(f)[0]
        groups.setdefault(base, []).append(f)

    for base_name, flist in groups.items():
        hea = any(f.endswith('.hea') for f in flist)
        mat = any(f.endswith('.mat') for f in flist)
        dat = any(f.endswith('.dat') for f in flist)
        if not hea or not (mat or dat):
            continue
        try:
            signal_data, fs = load_wfdb_file(target_dir, base_name)
            # keep raw values for UI (as list)
            mat_or_dat = next((f for f in flist if f.endswith('.mat') or f.endswith('.dat')), None)
            file_path = os.path.join(target_dir, mat_or_dat) if mat_or_dat else os.path.join(target_dir, base_name + '.hea')
            file_size = os.path.getsize(file_path)
            last_modified = os.path.getmtime(file_path)
            ecg_data.append({
                "filename": base_name,
                "signal": signal_data.astype(float).tolist(),
                "fs": fs,
                "file_size": file_size,
                "last_modified": last_modified,
                "file_type": "WFDB"
            })
        except Exception as e:
            logger.error(f"Error processing showcase file {base_name} in {target_dir}: {e}")

    return JSONResponse(content=ecg_data)

@app.put("/admin/rename-showcase-ecg/{old_filename}")
async def rename_showcase_ecg(old_filename: str, body: RenameRequest):
    new_name = body.new_filename
    new_name = sanitize_wfdb_name(new_name)
    old_base, _ = os.path.splitext(old_filename)

    # Try both folders
    candidate_folders = [PREDICTION_SHOWCASE_DIR, VIEWER_SHOWCASE_DIR]
    target_folder = None
    for folder in candidate_folders:
        old_mat = os.path.join(folder, f"{old_base}.dat")
        old_hea = os.path.join(folder, f"{old_base}.hea")
        if os.path.exists(old_mat) and os.path.exists(old_hea):
            target_folder = folder
            break

    if not target_folder:
        raise HTTPException(status_code=404, detail=f"Files for '{old_filename}' not found in any showcase folder")

    # Rename files
    new_mat = os.path.join(target_folder, f"{new_name}.dat")
    new_hea = os.path.join(target_folder, f"{new_name}.hea")
    os.rename(old_mat, new_mat)
    os.rename(old_hea, new_hea)

    # Update internal .hea references
    with open(new_hea, "r") as f:
        lines = f.readlines()
    updated_lines = [line.replace(old_base, new_name) for line in lines]
    with open(new_hea, "w") as f:
        f.writelines(updated_lines)

    return {"message": f"Renamed {old_base} to {new_name} successfully in folder '{os.path.basename(target_folder)}'"}


@app.delete("/admin/delete-showcase-ecg/{filename}")
async def delete_showcase_ecg(filename: str, folder_type: str = "prediction"):
    target_dir = PREDICTION_SHOWCASE_DIR if folder_type == "prediction" else VIEWER_SHOWCASE_DIR if folder_type == "viewer" else None
    if target_dir is None:
        raise HTTPException(status_code=400, detail="Invalid folder_type. Must be 'prediction' or 'viewer'.")
    if not os.path.exists(target_dir):
        raise HTTPException(status_code=404, detail="Showcase directory not found")

    base = os.path.splitext(filename)[0]
    to_delete = [f for f in os.listdir(target_dir) if os.path.splitext(f)[0] == base]
    if not to_delete:
        raise HTTPException(status_code=404, detail=f"Files for {filename} not found in {folder_type} showcase")

    for f in to_delete:
        try:
            os.remove(os.path.join(target_dir, f))
        except Exception as e:
            logger.error(f"Delete failed for {f}: {e}")
    return JSONResponse(content={"message": f"Files for {filename} deleted successfully from {folder_type} showcase"})

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
