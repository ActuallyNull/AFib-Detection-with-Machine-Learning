import React from 'react';
import PredictionView from './PredictionView';

const PredictionPage = () => {
  return (
    <div className="container mx-auto p-4">
      {/* Main Title and Tagline */}
      <div className="text-center mb-8">
        <h2 className="text-4xl font-extrabold text-gray-900 dark:text-white mb-2">
          CardioScan AI
        </h2>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          Advanced ECG Analysis for Arrhythmia Detection
        </p>
      </div>

      {/* "ECG Prediction" heading */}
      <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">ECG Prediction</h1>
      <PredictionView />
    </div>
  );
};

export default PredictionPage;
