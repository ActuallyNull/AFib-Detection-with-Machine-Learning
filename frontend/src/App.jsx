import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import axios from 'axios';
import Header from './components/Header';
import ECGUploader from './components/ECGUploader';
import ECGDisplayCard from './components/ECGDisplayCard';
import AdminPanel from './components/AdminPanel';
import { ECGProvider } from './context/ECGContext';

const MainPage = () => {
    const [showcaseECGs, setShowcaseECGs] = useState([]);

    useEffect(() => {
        const fetchShowcaseECGs = async () => {
            try {
                const response = await axios.get('http://localhost:8000/showcase-ecgs');
                setShowcaseECGs(response.data);
            } catch (error) {
                console.error('Error fetching showcase ECGs:', error);
            }
        };
        fetchShowcaseECGs();
    }, []);

    return (
        <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
                {showcaseECGs.slice(0, 3).map((ecg) => (
                    <ECGDisplayCard key={ecg.filename} ecg={ecg} />
                ))}
            </div>
            <ECGUploader />
        </>
    );
};

function App() {
    return (
        <ECGProvider>
            <div className="min-h-screen bg-gray-50">
                <Header />
                <main className="container mx-auto px-4 py-8">
                    <Routes>
                        <Route path="/" element={<MainPage />} />
                        <Route path="/admin" element={<AdminPanel />} />
                    </Routes>
                </main>
            </div>
        </ECGProvider>
    );
}

export default App;