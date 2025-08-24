import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { FileText, Activity, Eye, Clock } from 'lucide-react'
import { ecgAPI } from '../services/api'

const ECGThumbnail = ({ fileData }) => {
  const [previewData, setPreviewData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const loadPreview = async () => {
      try {
        setLoading(true)
        let data
        if (fileData.type === 'wfdb') {
          // For WFDB files, use the WFDB-specific endpoint
          data = await ecgAPI.preprocessWFDB(fileData.wfdbFiles)
        } else {
          // For other files, use the regular endpoint
          data = await ecgAPI.preprocess(fileData.primaryFile)
        }
        setPreviewData(data)
      } catch (err) {
        console.error('Error loading preview:', err)
        setError('Failed to load preview')
      } finally {
        setLoading(false)
      }
    }

    loadPreview()
  }, [fileData])

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getFileIcon = (fileName) => {
    const ext = fileName.toLowerCase().split('.').pop()
    switch (ext) {
      case 'mat':
      case 'hea':
      case 'dat':
        return <Activity className="h-6 w-6 text-blue-500" />
      case 'csv':
        return <FileText className="h-6 w-6 text-green-500" />
      default:
        return <FileText className="h-6 w-6 text-gray-500" />
    }
  }

  const getFileType = () => {
    if (fileData.type === 'wfdb') {
      return 'WFDB'
    }
    const ext = fileData.name.toLowerCase().split('.').pop()
    return ext.toUpperCase()
  }

  if (loading) {
    return (
      <div className="ecg-thumbnail">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded mb-2"></div>
          <div className="h-32 bg-gray-200 rounded mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="ecg-thumbnail border-red-200 bg-red-50">
        <div className="flex items-center space-x-2 mb-2">
          {getFileIcon(fileData.name)}
          <span className="font-medium text-gray-900 truncate">{fileData.name}</span>
        </div>
        <div className="h-32 bg-red-100 rounded flex items-center justify-center">
          <span className="text-red-600 text-sm">Preview unavailable</span>
        </div>
        <div className="text-xs text-gray-500 mt-2">
          {formatFileSize(fileData.size)} • {formatDate(fileData.uploadedAt)}
        </div>
      </div>
    )
  }

  return (
    <div className="ecg-thumbnail group">
      {/* File Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          {getFileIcon(fileData.name)}
          <span className="font-medium text-gray-900 truncate">{fileData.name}</span>
        </div>
        <Link
          to={`/viewer/${fileData.id}`}
          className="opacity-0 group-hover:opacity-100 transition-opacity duration-200"
        >
          <Eye className="h-5 w-5 text-primary-500 hover:text-primary-600" />
        </Link>
      </div>

      {/* ECG Preview */}
      <div className="relative h-32 bg-gray-50 rounded-lg overflow-hidden mb-3">
        {previewData && previewData.signal && (
          <svg
            className="w-full h-full"
            viewBox={`0 0 ${previewData.signal.length} 100`}
            preserveAspectRatio="none"
          >
            <polyline
              points={previewData.signal.map((y, x) => `${x},${50 - y * 40}`).join(' ')}
              fill="none"
              stroke="#6c5ce7"
              strokeWidth="1"
            />
          </svg>
        )}
        
        {/* Overlay with file info */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <div className="absolute bottom-2 left-2 text-white text-xs">
            <div className="flex items-center space-x-1">
              <Clock className="h-3 w-3" />
              <span>{previewData?.duration || 'Unknown'}s</span>
            </div>
          </div>
        </div>
      </div>

      {/* File Info */}
      <div className="text-xs text-gray-500 space-y-1">
        <div className="flex justify-between">
          <span>Size: {formatFileSize(fileData.size)}</span>
          <span>Type: {getFileType()}</span>
        </div>
        <div className="flex items-center space-x-1">
          <Clock className="h-3 w-3" />
          <span>{formatDate(fileData.uploadedAt)}</span>
        </div>
        {fileData.type === 'wfdb' && fileData.wfdbFiles && (
          <div className="text-xs text-blue-600">
            {fileData.wfdbFiles.length} WFDB files
          </div>
        )}
      </div>

      {/* Quick Stats */}
      {previewData && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Samples:</span>
              <span className="ml-1 font-medium">{previewData.signal?.length || 0}</span>
            </div>
            <div>
              <span className="text-gray-500">Fs:</span>
              <span className="ml-1 font-medium">{previewData.fs || 'Unknown'} Hz</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ECGThumbnail

