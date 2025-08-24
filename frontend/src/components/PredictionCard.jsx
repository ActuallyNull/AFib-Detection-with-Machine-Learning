import React from 'react'
import { useECG } from '../context/ECGContext'
import { Heart, Activity, AlertCircle, CheckCircle } from 'lucide-react'

const PredictionCard = ({ fileId }) => {
  const { predictions } = useECG()
  const prediction = predictions[fileId]

  if (!prediction) {
    return (
      <div className="card bg-gray-50 border-gray-200">
        <div className="flex items-center space-x-2">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500"></div>
          <span className="text-sm text-gray-600">Processing...</span>
        </div>
      </div>
    )
  }

  if (prediction.error) {
    return (
      <div className="card bg-red-50 border-red-200">
        <div className="flex items-center space-x-2">
          <AlertCircle className="h-5 w-5 text-red-500" />
          <div>
            <h4 className="font-semibold text-red-800">Prediction Error</h4>
            <p className="text-sm text-red-700">{prediction.error}</p>
          </div>
        </div>
      </div>
    )
  }

  const getPredictionIcon = (label) => {
    switch (label?.toLowerCase()) {
      case 'afib':
      case 'atrial fibrillation':
        return <Heart className="h-5 w-5 text-red-500" />
      case 'normal':
      case 'sinus rhythm':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'other':
      case 'other arrhythmia':
        return <Activity className="h-5 w-5 text-orange-500" />
      default:
        return <Activity className="h-5 w-5 text-blue-500" />
    }
  }

  const getPredictionColor = (label) => {
    switch (label?.toLowerCase()) {
      case 'afib':
      case 'atrial fibrillation':
        return 'bg-red-50 border-red-200 text-red-800'
      case 'normal':
      case 'sinus rhythm':
        return 'bg-green-50 border-green-200 text-green-800'
      case 'other':
      case 'other arrhythmia':
        return 'bg-orange-50 border-orange-200 text-orange-800'
      default:
        return 'bg-blue-50 border-blue-200 text-blue-800'
    }
  }

  return (
    <div className={`card ${getPredictionColor(prediction.label)}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          {getPredictionIcon(prediction.label)}
          <h4 className="font-semibold">AI Prediction</h4>
        </div>
      </div>
      
      <div className="space-y-2">
        <div>
          <span className="text-sm font-medium">Label: </span>
          <span className="text-sm font-bold">{prediction.label || 'Unknown'}</span>
        </div>
        
        {prediction.processing_time && (
          <div className="text-xs text-gray-600 mt-2 pt-2 border-t border-gray-200">
            Processing time: {prediction.processing_time.toFixed(2)}s
          </div>
        )}
      </div>
    </div>
  )
}

export default PredictionCard