import React, { useState, useEffect, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { fabric } from 'fabric'
import { 
  ArrowLeft, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Download, 
  Edit3, 
  Save,
  X,
  Heart,
  Activity,
  Undo,
  Redo
} from 'lucide-react'
import { useECG } from '../context/ECGContext'
import { ecgAPI } from '../services/api'

const ECGViewer = () => {
  const { id } = useParams()
  const { ecgFiles, predictions } = useECG()
  const [ecgData, setEcgData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [plotConfig, setPlotConfig] = useState({
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    modeBarButtonsToAdd: [
      {
        name: 'Reset View',
        icon: 'reset',
        click: () => {
          // Reset view logic
        }
      }
    ]
  })
  const [annotationMode, setAnnotationMode] = useState(false)
  const [canvas, setCanvas] = useState(null)
  const [canvasHistory, setCanvasHistory] = useState([]);
  const [canvasRedoHistory, setCanvasRedoHistory] = useState([]);
  const plotRef = useRef(null)
  const canvasRef = useRef(null)

  const fileData = ecgFiles.find(f => f.id === id)
  const prediction = predictions[id]

  useEffect(() => {
    const loadECGData = async () => {
      if (!fileData) {
        setError('ECG file not found')
        setLoading(false)
        return
      }

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
        setEcgData(data)
      } catch (err) {
        console.error('Error loading ECG data:', err)
        setError('Failed to load ECG data')
      } finally {
        setLoading(false)
      }
    }

    loadECGData()
  }, [fileData])

  useEffect(() => {
    if (annotationMode && canvasRef.current && ecgData) {
      const fabricCanvas = new fabric.Canvas(canvasRef.current, {
        width: 800,
        height: 400,
        backgroundColor: 'transparent'
      });

      const saveState = () => {
        const canvasState = fabricCanvas.toJSON();
        setCanvasHistory(prev => [...prev, canvasState]);
        setCanvasRedoHistory([]); 
      };

      fabricCanvas.on({
        'object:added': saveState,
        'object:modified': saveState,
        'object:removed': saveState
      });

      setCanvas(fabricCanvas);
      setCanvasHistory([fabricCanvas.toJSON()]);

      return () => {
        fabricCanvas.off('object:added');
        fabricCanvas.off('object:modified');
        fabricCanvas.off('object:removed');
        fabricCanvas.dispose();
      };
    }
  }, [annotationMode, ecgData]);

  const handleZoomIn = () => {
    if (plotRef.current) {
      const update = {
        'xaxis.range': [plotRef.current.layout.xaxis.range[0] * 0.8, plotRef.current.layout.xaxis.range[1] * 0.8]
      }
      plotRef.current.relayout(update)
    }
  }

  const handleZoomOut = () => {
    if (plotRef.current) {
      const update = {
        'xaxis.range': [plotRef.current.layout.xaxis.range[0] * 1.2, plotRef.current.layout.xaxis.range[1] * 1.2]
      }
      plotRef.current.relayout(update)
    }
  }

  const handleResetView = () => {
    if (plotRef.current && ecgData) {
      const update = {
        'xaxis.range': [0, ecgData.signal.length],
        'yaxis.range': [Math.min(...ecgData.signal) - 0.1, Math.max(...ecgData.signal) + 0.1]
      }
      plotRef.current.relayout(update)
    }
  }

  const addAnnotation = (type) => {
    if (!canvas) return

    let annotation
    switch (type) {
      case 'text':
        annotation = new fabric.Text('Annotation', {
          left: 100,
          top: 100,
          fontSize: 16,
          fill: '#ff0000'
        })
        break
      case 'line':
        annotation = new fabric.Line([50, 50, 200, 50], {
          stroke: '#ff0000',
          strokeWidth: 2
        })
        break
      case 'rect':
        annotation = new fabric.Rect({
          left: 100,
          top: 100,
          width: 100,
          height: 50,
          fill: 'transparent',
          stroke: '#ff0000',
          strokeWidth: 2
        })
        break
      default:
        return
    }

    canvas.add(annotation)
    canvas.renderAll()
  }

  const handleUndo = () => {
    if (canvasHistory.length > 1) {
      const lastState = canvasHistory[canvasHistory.length - 1];
      const prevState = canvasHistory[canvasHistory.length - 2];
      
      setCanvasRedoHistory(prev => [lastState, ...prev]);
      
      canvas.loadFromJSON(prevState, () => {
        canvas.renderAll();
        setCanvasHistory(prev => prev.slice(0, prev.length - 1));
      });
    }
  };

  const handleRedo = () => {
    if (canvasRedoHistory.length > 0) {
      const nextState = canvasRedoHistory[0];
      
      setCanvasHistory(prev => [...prev, nextState]);
      
      canvas.loadFromJSON(nextState, () => {
        canvas.renderAll();
        setCanvasRedoHistory(prev => prev.slice(1));
      });
    }
  };

  const saveAnnotations = () => {
    if (!canvas) return
    
    const annotations = canvas.getObjects().map(obj => ({
      type: obj.type,
      left: obj.left,
      top: obj.top,
      width: obj.width,
      height: obj.height,
      text: obj.text,
      stroke: obj.stroke,
      fill: obj.fill
    }))

    console.log('Saving annotations:', annotations)
    // Here you would typically save to backend
    alert('Annotations saved!')
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading ECG data...</p>
        </div>
      </div>
    )
  }

  if (error || !fileData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <X className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Error</h2>
          <p className="text-gray-600 mb-4">{error || 'ECG file not found'}</p>
          <Link to="/" className="btn-primary">
            Back to Upload
          </Link>
        </div>
      </div>
    )
  }

  const plotData = ecgData ? [
    {
      x: Array.from({ length: ecgData.signal.length }, (_, i) => i),
      y: ecgData.signal,
      type: 'scatter',
      mode: 'lines',
      line: {
        color: '#6c5ce7',
        width: 1
      },
      name: 'ECG Signal'
    }
  ] : []

  const plotLayout = {
    title: {
      text: fileData.name,
      font: { size: 18 }
    },
    xaxis: {
      title: 'Samples',
      showgrid: true,
      gridcolor: '#f0f0f0'
    },
    yaxis: {
      title: 'Amplitude',
      showgrid: true,
      gridcolor: '#f0f0f0'
    },
    margin: { l: 60, r: 30, t: 60, b: 60 },
    height: 500,
    showlegend: false,
    hovermode: 'x unified'
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link to="/" className="flex items-center space-x-2 text-gray-600 hover:text-gray-900">
                <ArrowLeft className="h-5 w-5" />
                <span>Back to Upload</span>
              </Link>
              <div className="h-6 w-px bg-gray-300"></div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">{fileData.name}</h1>
                <p className="text-sm text-gray-600">
                  {ecgData ? `${ecgData.signal.length} samples • ${ecgData.fs || 'Unknown'} Hz` : 'Loading...'}
                </p>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setAnnotationMode(!annotationMode)}
                className={`p-2 rounded-lg border ${
                  annotationMode 
                    ? 'bg-primary-500 text-white border-primary-500' 
                    : 'bg-white text-gray-600 border-gray-300 hover:border-primary-500'
                }`}
                title="Toggle Annotation Mode"
              >
                <Edit3 className="h-4 w-4" />
              </button>
              
              <button
                onClick={handleZoomIn}
                className="p-2 rounded-lg border border-gray-300 bg-white text-gray-600 hover:border-primary-500"
                title="Zoom In"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
              
              <button
                onClick={handleZoomOut}
                className="p-2 rounded-lg border border-gray-300 bg-white text-gray-600 hover:border-primary-500"
                title="Zoom Out"
              >
                <ZoomOut className="h-4 w-4" />
              </button>
              
              <button
                onClick={handleResetView}
                className="p-2 rounded-lg border border-gray-300 bg-white text-gray-600 hover:border-primary-500"
                title="Reset View"
              >
                <RotateCcw className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Main ECG Plot */}
          <div className="lg:col-span-3">
            <div className="card">
              <div className="relative">
                {annotationMode && (
                  <div className="absolute top-4 left-4 z-10 flex space-x-2">
                    <button
                      onClick={() => addAnnotation('text')}
                      className="px-3 py-1 bg-blue-500 text-white rounded text-sm"
                    >
                      Add Text
                    </button>
                    <button
                      onClick={() => addAnnotation('line')}
                      className="px-3 py-1 bg-green-500 text-white rounded text-sm"
                    >
                      Add Line
                    </button>
                    <button
                      onClick={() => addAnnotation('rect')}
                      className="px-3 py-1 bg-orange-500 text-white rounded text-sm"
                    >
                      Add Rectangle
                    </button>
                    <button
                      onClick={saveAnnotations}
                      className="px-3 py-1 bg-primary-500 text-white rounded text-sm"
                    >
                      <Save className="h-3 w-3 inline mr-1" />
                      Save
                    </button>
                    <button
                      onClick={handleUndo}
                      className="px-3 py-1 bg-gray-500 text-white rounded text-sm"
                      disabled={canvasHistory.length <= 1}
                    >
                      <Undo className="h-3 w-3 inline mr-1" />
                      Undo
                    </button>
                    <button
                      onClick={handleRedo}
                      className="px-3 py-1 bg-gray-500 text-white rounded text-sm"
                      disabled={canvasRedoHistory.length === 0}
                    >
                      <Redo className="h-3 w-3 inline mr-1" />
                      Redo
                    </button>
                  </div>
                )}
                
                <Plot
                  ref={plotRef}
                  data={plotData}
                  layout={plotLayout}
                  config={plotConfig}
                  style={{ width: '100%', height: '500px' }}
                  useResizeHandler={true}
                />
                
                {annotationMode && (
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 pointer-events-none"
                    style={{ width: '100%', height: '500px' }}
                  />
                )}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* File Info */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">File Information</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Name:</span>
                  <span className="font-medium">{fileData.name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Size:</span>
                  <span className="font-medium">
                    {(fileData.size / 1024).toFixed(1)} KB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Type:</span>
                  <span className="font-medium">{fileData.name.split('.').pop().toUpperCase()}</span>
                </div>
                {ecgData && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Samples:</span>
                      <span className="font-medium">{ecgData.signal.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Duration:</span>
                      <span className="font-medium">
                        {(ecgData.signal.length / (ecgData.fs || 300)).toFixed(1)}s
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Sampling Rate:</span>
                      <span className="font-medium">{ecgData.fs || 'Unknown'} Hz</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* AI Prediction */}
            {prediction && (
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Prediction</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {prediction.label?.toLowerCase() === 'afib' ? (
                        <Heart className="h-5 w-5 text-red-500" />
                      ) : (
                        <Activity className="h-5 w-5 text-green-500" />
                      )}
                      <span className="font-medium">{prediction.label || 'Unknown'}</span>
                    </div>
                    <span className="text-sm font-bold text-primary-600">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  {prediction.class_probabilities && (
                    <div className="space-y-2">
                      {Object.entries(prediction.class_probabilities).map(([className, prob]) => (
                        <div key={className} className="flex justify-between text-sm">
                          <span className="capitalize">{className}:</span>
                          <span className="font-medium">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Controls Info */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Controls</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <div>• Scroll to zoom in/out</div>
                <div>• Drag to pan</div>
                <div>• Double-click to reset view</div>
                <div>• Hover for signal values</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ECGViewer