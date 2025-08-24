# ECG Classification AI - React + FastAPI

A modern web application for AI-powered ECG classification and arrhythmia detection, built with React frontend and FastAPI backend.

## 🚀 Features

### Frontend (React)
- **Drag & Drop Upload**: Support for WFDB (.mat, .hea), CSV, and TXT files
- **Interactive ECG Viewer**: Fullscreen viewer with Plotly.js integration
- **Zoom & Scroll**: Advanced navigation controls for ECG analysis
- **Annotation Support**: Draw on ECG plots using Fabric.js
- **Real-time Predictions**: Instant AI predictions with confidence scores
- **Responsive Design**: Modern UI with Tailwind CSS

### Backend (FastAPI)
- **RESTful API**: Clean endpoints for ECG processing
- **Multi-format Support**: WFDB, CSV, TXT file processing
- **Model Serving**: TensorFlow/Keras model integration
- **Batch Processing**: Handle multiple files simultaneously
- **Health Monitoring**: Built-in health checks and logging

## 🏗️ Architecture

```
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── context/         # State management
│   │   ├── services/        # API services
│   │   └── ...
│   ├── Dockerfile
│   └── nginx.conf
├── backend/                  # FastAPI application
│   ├── main.py             # Main API server
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile
├── model.keras             # Pre-trained ECG model
├── docker-compose.yml      # Multi-service orchestration
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.9+ (for local development)

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd actualibproject-deployment-reactfastapi
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development

#### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

#### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 File Structure

### Frontend Components
- `ECGUploader`: Main upload interface with drag & drop
- `ECGThumbnail`: Preview cards for uploaded files
- `ECGViewer`: Fullscreen ECG viewer with annotations
- `PredictionCard`: AI prediction display
- `Header`: Navigation and branding

### Backend Endpoints
- `POST /predict`: ECG classification prediction
- `POST /preprocess`: Signal preprocessing for visualization
- `POST /batch-predict`: Multiple file processing
- `GET /health`: Health check
- `GET /model-info`: Model information

## 🔧 Configuration

### Environment Variables

#### Frontend (.env)
```env
VITE_API_URL=http://localhost:8000
```

#### Backend
```env
PYTHONPATH=/app
```

### Model Configuration
The application expects a `model.keras` file in the root directory. The model should:
- Accept input shape: `(None, 3000, 1)`
- Output 3 classes: `['AFib', 'Normal', 'Other']`
- Return softmax probabilities

## 📊 API Usage

### Single File Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ecg_data.mat"
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@file1.mat" \
  -F "files=@file2.csv"
```

### Preprocess for Visualization
```bash
curl -X POST "http://localhost:8000/preprocess" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ecg_data.mat"
```

## 🎨 Customization

### Styling
The frontend uses Tailwind CSS. Customize colors and styling in:
- `frontend/tailwind.config.js`
- `frontend/src/index.css`

### Model Integration
To use a different model:
1. Replace `model.keras` with your model file
2. Update `class_names` in `backend/main.py`
3. Adjust preprocessing in `preprocess_ecg()` function

### Adding New File Formats
Extend file support by adding new loaders in `backend/main.py`:
```python
def load_new_format(file_content: bytes):
    # Implementation for new format
    pass
```

## 🚀 Deployment

### Production Deployment
1. **Update environment variables**
2. **Configure nginx for production**
3. **Set up SSL certificates**
4. **Deploy with Docker Compose**

### Cloud Deployment
- **AWS**: Use ECS or EKS
- **Google Cloud**: Use Cloud Run or GKE
- **Azure**: Use Container Instances or AKS

## 🧪 Testing

### Frontend Tests
```bash
cd frontend
npm test
```

### Backend Tests
```bash
cd backend
python -m pytest
```

### API Tests
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test model info
curl http://localhost:8000/model-info
```

## 📈 Performance

### Optimization Tips
- **Model Caching**: Model is loaded once at startup
- **File Processing**: Efficient WFDB and CSV parsing
- **Frontend**: Lazy loading and code splitting
- **Caching**: Static asset caching with nginx

### Monitoring
- Health checks for both services
- Structured logging
- Performance metrics via FastAPI

## 🔒 Security

### Best Practices
- CORS configuration
- Input validation
- File type restrictions
- Rate limiting (can be added)
- Authentication (can be added)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with details

## 🔄 Migration from Streamlit

This React + FastAPI version provides:
- **Better Performance**: Faster loading and interactions
- **Enhanced UX**: Modern drag & drop interface
- **Scalability**: Microservices architecture
- **Extensibility**: Easy to add new features
- **Production Ready**: Docker deployment and monitoring
