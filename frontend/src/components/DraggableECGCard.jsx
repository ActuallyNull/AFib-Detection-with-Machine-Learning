import React from 'react';
import { useDrag } from 'react-dnd';
import Plot from 'react-plotly.js';

const ItemTypes = {
  ECG_CARD: 'ecg_card',
};

const DraggableECGCard = ({ ecg }) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: ItemTypes.ECG_CARD,
    item: { ecg },
    collect: (monitor) => ({
      isDragging: !!monitor.isDragging(),
    }),
  }));

  if (!ecg) {
    return null;
  }

  const { filename, signal, fs } = ecg;
  const time = Array.from({ length: signal.length }, (_, i) => i / fs);

  return (
    <div
      ref={drag}
      className="w-full border p-4 rounded-lg shadow-sm bg-white dark:bg-gray-700 dark:border-gray-600"
      style={{
        cursor: 'move',
        opacity: isDragging ? 0.5 : 1,
      }}
    >
      <h3 className="font-semibold text-lg mb-2 dark:text-white">{filename}</h3>
      <div className="w-full h-64">
        <Plot
          data={[
            {
              x: time,
              y: signal,
              type: 'scatter',
              mode: 'lines',
              marker: { color: '#1f77b4' },
            },
          ]}
          layout={{
            autosize: true,
            margin: { l: 40, r: 40, b: 40, t: 40 },
            xaxis: { title: 'Time (s)' },
            yaxis: { title: 'Amplitude' },
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

export default DraggableECGCard;