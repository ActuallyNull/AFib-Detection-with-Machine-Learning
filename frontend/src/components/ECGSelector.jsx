import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const ECGSelector = () => {
    const [ecgs, setEcgs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchEcgs = async () => {
            try {
                const response = await api.get('/showcase-ecgs?folder_type=viewer');
                setEcgs(response.data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchEcgs();
    }, []);

    if (loading) {
        return <div className="text-center p-8">Loading...</div>;
    }

    if (error) {
        return <div className="text-center p-8 text-red-500">Error: {error}</div>;
    }

    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold mb-4 text-gray-800 dark:text-white">Select an ECG to View</h1>
            {ecgs.length === 0 ? (
                <p className="text-gray-600 dark:text-gray-400">No ECGs found in the viewer showcase.</p>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {ecgs.map((ecg) => (
                        <div key={ecg.filename} className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-4 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                            <Link to={`/view-ecg/${ecg.filename}`} className="text-lg font-semibold text-blue-600 hover:underline">
                                {ecg.filename}
                            </Link>
                            <p className="text-gray-600 dark:text-gray-400">Sampling Rate: {ecg.fs} Hz</p>
                            <p className="text-gray-600 dark:text-gray-400">File Size: {ecg.file_size} bytes</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default ECGSelector;
