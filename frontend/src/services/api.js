import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth tokens if needed
api.interceptors.request.use(
  (config) => {
    // Add any auth headers here if needed
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export const ecgAPI = {
  // Upload and predict ECG
  predict: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Upload and predict WFDB files (multiple files)
  predictWFDB: async (files) => {
    const formData = new FormData()
    
    // Add all WFDB files to the form data
    files.forEach(file => {
      formData.append('files', file)
    })
    
    const response = await api.post('/predict-wfdb', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Preprocess ECG for visualization
  preprocess: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await api.post('/preprocess', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Preprocess WFDB files for visualization
  preprocessWFDB: async (files) => {
    const formData = new FormData()
    
    // Add all WFDB files to the form data
    files.forEach(file => {
      formData.append('files', file)
    })
    
    const response = await api.post('/preprocess-wfdb', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Get health status
  health: async () => {
    const response = await api.get('/health')
    return response.data
  },

  // Get model info
  modelInfo: async () => {
    const response = await api.get('/model-info')
    return response.data
  }
}

export default api

