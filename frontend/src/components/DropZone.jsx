import React from 'react';
import { useDrop } from 'react-dnd';
import DraggableECGCard from './DraggableECGCard';

const ItemTypes = {
  ECG_CARD: 'ecg_card',
};

const DropZone = ({ droppedECG, onDrop }) => {
  const [{ isOver }, drop] = useDrop(() => ({
    accept: ItemTypes.ECG_CARD,
    drop: (item) => onDrop(item.ecg),
    collect: (monitor) => ({
      isOver: !!monitor.isOver(),
    }),
  }));

  return (
    <div
      ref={drop}
      className={`w-full min-h-[288px] border-2 border-dashed rounded-lg flex items-center justify-center transition-colors ${isOver ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/50' : 'border-gray-300 dark:border-gray-600'}`}>
      {droppedECG ? (
        <DraggableECGCard ecg={droppedECG} />
      ) : (
        <div className="p-4 w-full h-full flex flex-col items-center justify-center space-y-4">
          <div className="w-3/4 h-6 bg-gray-300 dark:bg-gray-600 rounded animate-pulse"></div> {/* Filename placeholder */}
          <div className="w-full h-48 bg-gray-300 dark:bg-gray-600 rounded animate-pulse"></div> {/* Plot placeholder */}
          <p className="text-sm text-gray-500 dark:text-gray-400">Drag & Drop ECG here</p>
        </div>
      )}
    </div>
  );
};

export default DropZone;
