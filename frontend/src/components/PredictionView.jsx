import React, { useState, useEffect } from 'react';
import DraggableECGCard from './DraggableECGCard';
import PredictionControl from './PredictionControl';
import axios from 'axios';

const PredictionView = () => {
  const [ecgs, setEcgs] = useState([]);
  const [droppedECGFilename, setDroppedECGFilename] = useState(null);

  useEffect(() => {
    const fetchShowcaseECGs = async () => {
      try {
        const response = await axios.get('http://localhost:8000/showcase-ecgs?folder_type=prediction');
        setEcgs(response.data);
      } catch (error) {
        console.error('Error fetching showcase ECGs:', error);
      }
    };
    fetchShowcaseECGs();
  }, []);

  const handleDrop = (ecg) => {
    setDroppedECGFilename(ecg.filename);
  };

  const showcaseECGs = ecgs.filter(ecg => ecg.filename !== droppedECGFilename);
  const droppedECG = ecgs.find(ecg => ecg.filename === droppedECGFilename);

  return (
    <div>
      <h2 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">Showcase ECGs</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {showcaseECGs.map((ecg) => (
          <DraggableECGCard key={ecg.filename} ecg={ecg} />
        ))}
        {droppedECG && <div className="border-2 border-dashed rounded-lg" />}
      </div>
      <PredictionControl droppedECG={droppedECG} onDrop={handleDrop} />
    </div>
  );
};

export default PredictionView;