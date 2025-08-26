import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { Fullscreen, Minimize, PenTool, Eraser, X, Undo, Redo } from 'lucide-react';

const ECGImageViewer = () => {
  const { filename } = useParams();
  const ecgCanvasRef = useRef(null);
  const annotationCanvasRef = useRef(null);
  const previewCanvasRef = useRef(null); // New ref for preview canvas

  const [ecgData, setEcgData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingMode, setDrawingMode] = useState(null); // 'pen', 'eraser'
  const [annotations, setAnnotations] = useState([]);
  const [currentAnnotation, setCurrentAnnotation] = useState(null);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [annotationHistory, setAnnotationHistory] = useState([[]]);
  const [historyIndex, setHistoryIndex] = useState(0);

  // New state for pencil options
  const [pencilColor, setPencilColor] = useState('red');
  const [pencilThickness, setPencilThickness] = useState(2);

  // State for mouse position for preview circle
  const [mousePos, setMousePos] = useState({ x: -1, y: -1 }); // -1 to hide initially

  // Function to draw only the ECG signal and grid
  const drawECG = useCallback((ctx, signal, fs, width, height) => {
    if (!signal || signal.length === 0) return;

    ctx.clearRect(0, 0, width, height);

    // Draw grid (simplified for now)
    ctx.strokeStyle = 'rgba(200, 0, 0, 0.3)'; // Light red for grid
    ctx.lineWidth = 0.5;

    const mV_per_mm = 0.1; // 1mm = 0.1mV (standard ECG)
    const ms_per_mm = 40;  // 1mm = 40ms (standard ECG)
    const pixels_per_mm = 3.78; // Assuming 96 DPI, 1 inch = 25.4mm, 96px/inch => 96/25.4 = 3.78 px/mm

    const grid_spacing_mm_small = 1; // 1mm grid
    const grid_spacing_mm_large = 5; // 5mm grid

    const px_small = grid_spacing_mm_small * pixels_per_mm;
    const px_large = grid_spacing_mm_large * pixels_per_mm;

    // Vertical lines (time)
    for (let x = 0; x < width; x += px_small) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    // Horizontal lines (amplitude)
    for (let y = 0; y < height; y += px_small) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw larger grid lines
    ctx.strokeStyle = 'rgba(200, 0, 0, 0.5)'; // Darker red for large grid
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += px_large) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += px_large) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw ECG signal
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const max_val = Math.max(...signal);
    const min_val = Math.min(...signal);
    const range = max_val - min_val;

    // Scale signal to fit canvas height
    const scaleY = height / (range * 1.2); // 1.2 to add some margin
    const offsetY = height / 2 - (max_val + min_val) / 2 * scaleY; // Center the signal vertically

    const points_per_pixel = signal.length / width;

    for (let i = 0; i < width; i++) {
      const signal_index = Math.floor(i * points_per_pixel);
      const x = i;
      const y = height - (signal[signal_index] - min_val) * scaleY; // Invert Y for canvas

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }, []); // No dependency on annotations anymore

  // Function to draw only annotations
  const drawAnnotations = useCallback((ctx, currentAnnotations, width, height) => {
    ctx.clearRect(0, 0, width, height); // Clear annotation canvas

    currentAnnotations.forEach(ann => {
      ctx.save(); // Save the current context state
      ctx.strokeStyle = ann.color; // Use annotation's color
      ctx.lineWidth = ann.thickness; // Use annotation's thickness
      ctx.globalCompositeOperation = ann.mode === 'pen' ? 'source-over' : 'destination-out'; // Eraser effect
      ctx.beginPath();
      ann.points.forEach((point, index) => {
        if (index === 0) {
          ctx.moveTo(point.x, point.y);
        } else {
          ctx.lineTo(point.x, point.y);
        }
      });
      ctx.stroke();
      ctx.restore(); // Restore the context state
    });
  }, []);

  // Function to draw the preview circle
  const drawPreviewCircle = useCallback((ctx, x, y, radius, color) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clear previous preview

    if (x === -1 || y === -1 || !drawingMode) return; // Hide if mouse leaves or no tool selected

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }, [drawingMode]);

  useEffect(() => {
    const fetchECG = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:8000/showcase-ecgs?folder_type=viewer');
        const foundEcg = response.data.find(ecg => ecg.filename === filename);
        if (foundEcg) {
          setEcgData(foundEcg);
        } else {
          setError('ECG not found.');
        }
      } catch (err) {
        console.error('Error fetching ECG:', err);
        setError('Failed to load ECG data.');
      } finally {
        setLoading(false);
      }
    };
    fetchECG();
  }, [filename]);

  // Effect to draw ECG signal (runs once when ecgData loads and on resize)
  useEffect(() => {
    const ecgCanvas = ecgCanvasRef.current;
    const annotationCanvas = annotationCanvasRef.current;
    const previewCanvas = previewCanvasRef.current; // Get preview canvas

    if (!ecgCanvas || !annotationCanvas || !previewCanvas || !ecgData) return;

    const ecgCtx = ecgCanvas.getContext('2d');
    const annotationCtx = annotationCanvas.getContext('2d');
    const { signal, fs } = ecgData;

    const resizeCanvases = () => {
      ecgCanvas.width = ecgCanvas.offsetWidth;
      ecgCanvas.height = ecgCanvas.offsetHeight;
      annotationCanvas.width = annotationCanvas.offsetWidth;
      annotationCanvas.height = annotationCanvas.offsetHeight;
      previewCanvas.width = previewCanvas.offsetWidth; // Resize preview canvas
      previewCanvas.height = previewCanvas.offsetHeight; // Resize preview canvas

      drawECG(ecgCtx, signal, fs, ecgCanvas.width, ecgCanvas.height);
      // Redraw annotations on resize as well
      drawAnnotations(annotationCanvas.getContext('2d'), annotations, annotationCanvas.width, annotationCanvas.height);
    };

    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);

    return () => {
      window.removeEventListener('resize', resizeCanvases);
    };
  }, [ecgData, drawECG, drawAnnotations, annotations]);

  // Effect to draw annotations (runs when annotations or drawing state changes)
  useEffect(() => {
    const annotationCanvas = annotationCanvasRef.current;
    if (!annotationCanvas || !ecgData) return; // ecgData is needed for dimensions

    const annotationCtx = annotationCanvas.getContext('2d');
    drawAnnotations(annotationCtx, annotations, annotationCanvas.width, annotationCanvas.height);

    // Draw current annotation being drawn (interactive stroke)
    if (isDrawing && currentAnnotation) {
      annotationCtx.save();
      annotationCtx.strokeStyle = currentAnnotation.color;
      annotationCtx.lineWidth = currentAnnotation.thickness;
      annotationCtx.globalCompositeOperation = currentAnnotation.mode === 'pen' ? 'source-over' : 'destination-out';
      annotationCtx.beginPath();
      currentAnnotation.points.forEach((point, index) => {
        if (index === 0) {
          annotationCtx.moveTo(point.x, point.y);
        } else {
          annotationCtx.lineTo(point.x, point.y);
        }
      });
      annotationCtx.stroke();
      annotationCtx.restore();
    }
  }, [annotations, isDrawing, currentAnnotation, drawAnnotations, ecgData]);

  // Effect to draw preview circle
  useEffect(() => {
    const previewCanvas = previewCanvasRef.current;
    if (!previewCanvas) return;
    const ctx = previewCanvas.getContext('2d');

    let color = 'rgba(0, 0, 0, 0.5)'; // Default for eraser
    let radius = pencilThickness / 2;

    if (drawingMode === 'pen') {
      color = pencilColor;
      radius = pencilThickness / 2;
    } else if (drawingMode === 'eraser') {
      color = 'rgba(128, 128, 128, 0.5)'; // Grey for eraser preview
      radius = 10 / 2; // Eraser thickness is fixed at 10
    }

    drawPreviewCircle(ctx, mousePos.x, mousePos.y, radius, color);
  }, [mousePos, drawingMode, pencilThickness, pencilColor, drawPreviewCircle]);

  // Drawing functionality
  const startDrawing = ({ nativeEvent }) => {
    if (!drawingMode) return;
    setIsDrawing(true);
    const { offsetX, offsetY } = nativeEvent;
    setCurrentAnnotation({
      mode: drawingMode,
      color: drawingMode === 'pen' ? pencilColor : 'white', // Eraser color is always white
      thickness: drawingMode === 'pen' ? pencilThickness : 10, // Eraser thickness
      points: [{ x: offsetX, y: offsetY }]
    });
  };

  const draw = ({ nativeEvent }) => {
    if (!isDrawing || !currentAnnotation) return;
    const { offsetX, offsetY } = nativeEvent;
    setCurrentAnnotation(prev => ({
      ...prev,
      points: [...prev.points, { x: offsetX, y: offsetY }]
    }));
  };

  const endDrawing = () => {
    setIsDrawing(false);
    if (currentAnnotation && currentAnnotation.points.length > 1) {
      const newAnnotations = [...annotations, currentAnnotation];
      setAnnotations(newAnnotations);
      
      const newHistory = annotationHistory.slice(0, historyIndex + 1);
      setAnnotationHistory([...newHistory, newAnnotations]);
      setHistoryIndex(newHistory.length);
    }
    setCurrentAnnotation(null);
  };

  const handleUndo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      setAnnotations(annotationHistory[newIndex]);
    }
  };

  const handleRedo = () => {
    if (historyIndex < annotationHistory.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      setAnnotations(annotationHistory[newIndex]);
    }
  };

  // Fullscreen functionality
  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullScreen(true);
    } else {
      document.exitFullscreen();
      setIsFullScreen(false);
    }
  };

  return (
    <div className="container mx-auto p-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {ecgData && (
        <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">ECG Viewer: {ecgData.filename}</h1>
      )}

      <div className="flex justify-between items-center mb-4">
        <div className="flex space-x-2">
          {/* Color options */}
          {drawingMode === 'pen' && (
            <>
              {['red', 'blue', 'black'].map(color => (
                <button
                  key={color}
                  onClick={() => setPencilColor(color)}
                  className={`w-8 h-8 rounded-full border-2 ${pencilColor === color ? 'border-blue-500' : 'border-transparent'}`}
                  style={{ backgroundColor: color }}
                  title={`${color} pen`}
                ></button>
              ))}
              <label
                htmlFor="color-picker"
                className="w-8 h-8 rounded-full border-2"
                style={{ backgroundColor: pencilColor, cursor: 'pointer' }}
                title="Custom Color"
              ></label>
              <input
                id="color-picker"
                type="color"
                value={pencilColor}
                onChange={(e) => setPencilColor(e.target.value)}
                className="w-0 h-0 opacity-0 absolute"
              />
            </>
          )}

          {/* Thickness slider */}
          {drawingMode === 'pen' && (
            <input
              type="range"
              min="1"
              max="10"
              value={pencilThickness}
              onChange={(e) => setPencilThickness(Number(e.target.value))}
              className="w-24"
              title="Pencil Thickness"
            />
          )}

          <button
            onClick={() => setDrawingMode('pen')}
            className={`p-2 rounded-full ${drawingMode === 'pen' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200'} hover:bg-blue-600 dark:hover:bg-blue-500 transition-colors`}
            title="Pen Tool"
          >
            <PenTool size={20} />
          </button>
          <button
            onClick={() => setDrawingMode('eraser')}
            className={`p-2 rounded-full ${drawingMode === 'eraser' ? 'bg-red-500 text-white' : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200'} hover:bg-red-600 dark:hover:bg-red-500 transition-colors`}
            title="Eraser Tool"
          >
            <Eraser size={20} />
          </button>
          <button
            onClick={() => { 
              setAnnotations([]); 
              setDrawingMode(null);
              const newHistory = annotationHistory.slice(0, historyIndex + 1);
              setAnnotationHistory([...newHistory, []]);
              setHistoryIndex(newHistory.length);
            }}
            className="p-2 rounded-full bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            title="Clear All Annotations"
          >
            <X size={20} />
          </button>
          <button
            onClick={handleUndo}
            disabled={historyIndex === 0}
            className="p-2 rounded-full bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Undo"
          >
            <Undo size={20} />
          </button>
          <button
            onClick={handleRedo}
            disabled={historyIndex === annotationHistory.length - 1}
            className="p-2 rounded-full bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Redo"
          >
            <Redo size={20} />
          </button>
        </div>
        <button
          onClick={toggleFullScreen}
          className="p-2 rounded-full bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          title={isFullScreen ? "Exit Fullscreen" : "Fullscreen"}
        >
          {isFullScreen ? <Minimize size={20} /> : <Fullscreen size={20} />}
        </button>
      </div>

      <div className="relative w-full h-[600px] border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden">
        <canvas
          ref={ecgCanvasRef}
          className="absolute top-0 left-0 w-full h-full"
        />
        <canvas
          ref={annotationCanvasRef}
          className="absolute top-0 left-0 w-full h-full"
        />
        <canvas
          ref={previewCanvasRef} // New preview canvas
          className="absolute top-0 left-0 w-full h-full cursor-none" // Hide default cursor
          onMouseDown={startDrawing}
          onMouseMove={(e) => {
            setMousePos({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY });
            if (isDrawing) {
              draw(e);
            }
          }}
          onMouseUp={endDrawing}
          onMouseLeave={() => {
            endDrawing(); // End drawing if mouse leaves canvas
            setMousePos({ x: -1, y: -1 }); // Hide preview when mouse leaves
          }}
        />
      </div>
    </div>
  );
};

export default ECGImageViewer;