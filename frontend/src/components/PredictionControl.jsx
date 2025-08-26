import React, { useState } from 'react';
import DropZone from './DropZone';
import { Play, Loader } from 'lucide-react';
import axios from 'axios';

const PredictionControl = ({ droppedECG, onDrop }) => {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async () => {
    if (!droppedECG) {
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        filename: droppedECG.filename,
      });
      setPrediction(response.data.label);
    } catch (error) {
      console.error('Error making prediction:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getPredictionGlossyColor = (p) => {
    if (p === 'Normal') return 'from-green-400 to-green-600';
    if (p === 'AFib') return 'from-red-400 to-red-600';
    if (p === 'Other') return 'from-yellow-400 to-yellow-600';
    return 'from-gray-400 to-gray-600';
  };

  return (
    <div className="grid grid-cols-3 gap-4 items-center">
      <div className="col-span-1">
        <DropZone droppedECG={droppedECG} onDrop={onDrop} />
      </div>
      <div className="col-span-1 flex justify-center">
        <button
          onClick={handlePredict}
          disabled={!droppedECG || isLoading}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold rounded-full w-20 h-20 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? <Loader className="animate-spin" /> : <Play size={32} />}
        </button>
      </div>
      <div className="col-span-1">
        {prediction && (
          <div className={`p-4 rounded-lg text-white text-center bg-gradient-to-b ${getPredictionGlossyColor(prediction)} shadow-lg`}>
            <p className="text-lg font-semibold">Prediction:</p>
            <p className="text-2xl font-bold">{prediction}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionControl;
