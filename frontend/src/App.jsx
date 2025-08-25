import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import PredictionPage from './components/PredictionPage';
import AdminPanel from './components/AdminPanel';
import ECGImageViewer from './components/ECGImageViewer'; // Import ECGImageViewer
import ECGSelector from './components/ECGSelector';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { ECGProvider } from './context/ECGContext';
import { ThemeProvider } from './context/ThemeContext';

function App() {
    return (
        <ThemeProvider>
            <DndProvider backend={HTML5Backend}>
                <ECGProvider>
                    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
                        <Header />
                        <main className="container mx-auto px-4 py-8">
                            <Routes>
                                <Route path="/" element={<PredictionPage />} />
                                <Route path="/admin" element={<AdminPanel />} />
                                <Route path="/view-ecg" element={<ECGSelector />} />
                                <Route path="/view-ecg/:filename" element={<ECGImageViewer />} /> {/* New route */}
                            </Routes>
                        </main>
                    </div>
                </ECGProvider>
            </DndProvider>
        </ThemeProvider>
    );
}

export default App;