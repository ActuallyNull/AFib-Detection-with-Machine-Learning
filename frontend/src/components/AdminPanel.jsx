
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const AdminPanel = () => {
    const [files, setFiles] = useState([]);
    const [showcaseECGs, setShowcaseECGs] = useState([]);
    const [message, setMessage] = useState('');

    const fetchShowcaseECGs = async () => {
        try {
            const response = await axios.get('http://localhost:8000/showcase-ecgs');
            setShowcaseECGs(response.data);
        } catch (error) {
            console.error('Error fetching showcase ECGs:', error);
        }
    };

    useEffect(() => {
        fetchShowcaseECGs();
    }, []);

    const handleFileChange = (e) => {
        setFiles(e.target.files);
    };

    const handleUpload = async () => {
        if (files.length === 0) {
            setMessage('Please select files to upload.');
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        try {
            const response = await axios.post('http://localhost:8000/admin/upload-showcase-ecg', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setMessage(response.data.message);
            fetchShowcaseECGs();
        } catch (error) {
            setMessage('Error uploading files.');
            console.error('Error uploading files:', error);
        }
    };

    const handleDelete = async (filename) => {
        try {
            const response = await axios.delete(`http://localhost:8000/admin/delete-showcase-ecg/${filename}`);
            setMessage(response.data.message);
            fetchShowcaseECGs();
        } catch (error) {
            setMessage('Error deleting file.');
            console.error('Error deleting file:', error);
        }
    };

    return (
        <div className="container mx-auto p-4">
            <h1 className="text-2xl font-bold mb-4">Admin Panel</h1>
            <div className="mb-4">
                <h2 className="text-xl font-semibold mb-2">Upload Showcase ECGs</h2>
                <input type="file" multiple onChange={handleFileChange} className="mb-2" />
                <button onClick={handleUpload} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                    Upload
                </button>
                {message && <p className="mt-2">{message}</p>}
            </div>
            <div>
                <h2 className="text-xl font-semibold mb-2">Showcase ECGs</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {showcaseECGs.map((ecg) => (
                        <div key={ecg.filename} className="border p-4 rounded-lg">
                            <p className="font-semibold">{ecg.filename}</p>
                            <p>Sampling Rate: {ecg.fs}</p>
                            <p>File Size: {ecg.file_size} bytes</p>
                            <p>Last Modified: {new Date(ecg.last_modified * 1000).toLocaleString()}</p>
                            <button onClick={() => handleDelete(ecg.filename)} className="bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-2 rounded mt-2">
                                Delete
                            </button>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default AdminPanel;
