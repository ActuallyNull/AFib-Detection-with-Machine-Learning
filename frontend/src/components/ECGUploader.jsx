import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, Activity, AlertCircle, Heart } from 'lucide-react'
import { useECG } from '../context/ECGContext'
import { ecgAPI } from '../services/api'
import ECGThumbnail from './ECGThumbnail'
import PredictionCard from './PredictionCard'

const ECGUploader = () => {
  const { ecgFiles, addECGFiles, setPrediction, setLoading, setError, clearError } = useECG()
  const [dragOver, setDragOver] = useState(false)

  const validateWFDBFiles = (files) => {
    const wfdbFiles = files.filter(file => {
      const ext = file.name.toLowerCase().split('.').pop()
      return ext === 'hea' || ext === 'mat' || ext === 'dat'
    })

    if (wfdbFiles.length > 0) {
      // Group files by base name (without extension)
      const fileGroups = {}
      wfdbFiles.forEach(file => {
        const baseName = file.name.toLowerCase().replace(/\.(hea|mat|dat)$/, '')
        if (!fileGroups[baseName]) {
          fileGroups[baseName] = []
        }
        fileGroups[baseName].push(file)
      })

      // Check if each group has both .hea and .mat files
      const incompleteGroups = Object.entries(fileGroups).filter(([baseName, groupFiles]) => {
        const hasHea = groupFiles.some(f => f.name.toLowerCase().endsWith('.hea'))
        const hasMat = groupFiles.some(f => f.name.toLowerCase().endsWith('.mat'))
        return !hasHea || !hasMat
      })

      if (incompleteGroups.length > 0) {
        const missingFiles = incompleteGroups.map(([baseName]) => baseName).join(', ')
        throw new Error(`Incomplete WFDB files detected. The following records are missing required files: ${missingFiles}. Please upload both .hea and .mat files for each record.`)
      }
    }

    return files
  }

  const onDrop = useCallback(async (acceptedFiles) => {
    clearError()
    setLoading(true)
    
    try {
      // Validate WFDB files first
      const validFiles = validateWFDBFiles(acceptedFiles)
      
      // Filter for ECG files (WFDB, CSV, etc.)
      const ecgFiles = validFiles.filter(file => {
        const validExtensions = ['.mat', '.hea', '.csv', '.txt', '.dat']
        return validExtensions.some(ext => 
          file.name.toLowerCase().endsWith(ext)
        )
      })

      if (ecgFiles.length === 0) {
        throw new Error('No valid ECG files found. Please upload .mat, .hea, .csv, .txt, or .dat files.')
      }

      // Group WFDB files together for processing
      const wfdbGroups = {}
      const otherFiles = []

      ecgFiles.forEach(file => {
        const ext = file.name.toLowerCase().split('.').pop()
        if (ext === 'hea' || ext === 'mat' || ext === 'dat') {
          const baseName = file.name.toLowerCase().replace(/\.(hea|mat|dat)$/, '')
          if (!wfdbGroups[baseName]) {
            wfdbGroups[baseName] = []
          }
          wfdbGroups[baseName].push(file)
        } else {
          otherFiles.push(file)
        }
      })

      // Create file entries for WFDB groups (use .hea file as primary)
      const wfdbFileEntries = Object.entries(wfdbGroups).map(([baseName, files]) => {
        const heaFile = files.find(f => f.name.toLowerCase().endsWith('.hea'))
        const matFile = files.find(f => f.name.toLowerCase().endsWith('.mat'))
        
        return {
          id: `ecg-${Date.now()}-${baseName}`,
          primaryFile: heaFile, // Use .hea as primary for display
          wfdbFiles: files, // Store all WFDB files
          name: heaFile.name,
          size: files.reduce((sum, f) => sum + f.size, 0),
          type: 'wfdb',
          uploadedAt: new Date().toISOString(),
          baseName: baseName
        }
      })

      // Create file entries for other files
      const otherFileEntries = otherFiles.map((file, index) => ({
        id: `ecg-${Date.now()}-other-${index}`,
        primaryFile: file,
        wfdbFiles: null,
        name: file.name,
        size: file.size,
        type: file.type,
        uploadedAt: new Date().toISOString()
      }))

      const allFileEntries = [...wfdbFileEntries, ...otherFileEntries]
      addECGFiles(allFileEntries)

      // Process each file for prediction
      for (const fileData of allFileEntries) {
        try {
          let prediction
          if (fileData.type === 'wfdb') {
            // For WFDB files, we need to send all related files
            prediction = await ecgAPI.predictWFDB(fileData.wfdbFiles)
          } else {
            // For other files, send the single file
            prediction = await ecgAPI.predict(fileData.primaryFile)
          }
          setPrediction(fileData.id, prediction)
        } catch (error) {
          console.error(`Error processing ${fileData.name}:`, error)
          setPrediction(fileData.id, {
            error: error.response?.data?.detail || 'Failed to process file'
          })
        }
      }
    } catch (error) {
      setError(error.message)
    } finally {
      setLoading(false)
    }
  }, [addECGFiles, setPrediction, setLoading, setError, clearError])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.mat', '.hea', '.dat'],
      'text/csv': ['.csv'],
      'text/plain': ['.txt']
    },
    multiple: true
  })

  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          ECG Classification AI
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Upload your ECG files and get instant AI-powered arrhythmia detection
        </p>
      </div>

      {/* Upload Area */}
      <div className="card mb-8">
        <div
          {...getRootProps()}
          className={`upload-area ${isDragActive ? 'drag-over' : ''}`}
          onDragEnter={() => setDragOver(true)}
          onDragLeave={() => setDragOver(false)}
        >
          <input {...getInputProps()} />
          <Upload className="h-16 w-16 text-primary-500 mx-auto mb-4" />
          <h3 className="text-2xl font-semibold text-gray-900 mb-2">
            {isDragActive ? 'Drop your ECG files here' : 'Drag & drop ECG files here'}
          </h3>
          <p className="text-gray-600 mb-4">
            or click to browse files
          </p>
          <div className="flex items-center justify-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center space-x-1">
              <FileText className="h-4 w-4" />
              <span>WFDB (.hea + .mat)</span>
            </div>
            <div className="flex items-center space-x-1">
              <Activity className="h-4 w-4" />
              <span>CSV (.csv)</span>
            </div>
            <div className="flex items-center space-x-1">
              <FileText className="h-4 w-4" />
              <span>TXT (.txt)</span>
            </div>
          </div>
          <div className="mt-4 text-xs text-gray-500">
            <p>⚠️ For WFDB files: Upload both .hea and .mat files together</p>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {useECG().error && (
        <div className="card mb-8 bg-red-50 border-red-200">
          <div className="flex items-center space-x-3">
            <AlertCircle className="h-6 w-6 text-red-500" />
            <div>
              <h3 className="text-lg font-semibold text-red-800">Upload Error</h3>
              <p className="text-red-700">{useECG().error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {useECG().loading && (
        <div className="card mb-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Processing ECG files...</p>
        </div>
      )}

      {/* ECG Files Grid */}
      {ecgFiles.length > 0 && (
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Uploaded ECG Files ({ecgFiles.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {ecgFiles.map((fileData) => (
              <div key={fileData.id} className="space-y-4">
                <ECGThumbnail fileData={fileData} />
                <PredictionCard fileId={fileData.id} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Information Section */}
      <div className="card mt-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">About ECG Classification</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="bg-red-100 rounded-full p-4 w-16 h-16 mx-auto mb-3 flex items-center justify-center">
              <Heart className="h-8 w-8 text-red-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">AFib Detection</h3>
            <p className="text-gray-600 text-sm">
              Detect atrial fibrillation and other arrhythmias with high accuracy
            </p>
          </div>
          <div className="text-center">
            <div className="bg-green-100 rounded-full p-4 w-16 h-16 mx-auto mb-3 flex items-center justify-center">
              <Activity className="h-8 w-8 text-green-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Real-time Analysis</h3>
            <p className="text-gray-600 text-sm">
              Get instant predictions with confidence scores
            </p>
          </div>
          <div className="text-center">
            <div className="bg-blue-100 rounded-full p-4 w-16 h-16 mx-auto mb-3 flex items-center justify-center">
              <FileText className="h-8 w-8 text-blue-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Multiple Formats</h3>
            <p className="text-gray-600 text-sm">
              Support for WFDB, CSV, and other ECG file formats
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ECGUploader

