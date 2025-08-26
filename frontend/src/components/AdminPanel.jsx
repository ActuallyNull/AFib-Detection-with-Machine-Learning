import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Trash2, Pencil } from 'lucide-react';

const AdminPanel = () => {
  const [predictionShowcaseECGs, setPredictionShowcaseECGs] = useState([]);
  const [viewerShowcaseECGs, setViewerShowcaseECGs] = useState([]);
  const [selectedPredictionFiles, setSelectedPredictionFiles] = useState([]);
  const [selectedViewerFiles, setSelectedViewerFiles] = useState([]);
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetchPredictionShowcaseECGs();
    fetchViewerShowcaseECGs();
  }, []);

  const fetchPredictionShowcaseECGs = async () => {
    try {
      const response = await axios.get('http://localhost:8000/showcase-ecgs?folder_type=prediction');
      setPredictionShowcaseECGs(response.data);
    } catch (error) {
      console.error('Error fetching prediction showcase ECGs:', error);
      setMessage('Error fetching prediction showcase ECGs.');
    }
  };

  const fetchViewerShowcaseECGs = async () => {
    try {
      const response = await axios.get('http://localhost:8000/showcase-ecgs?folder_type=viewer');
      setViewerShowcaseECGs(response.data);
    } catch (error) {
      console.error('Error fetching viewer showcase ECGs:', error);
      setMessage('Error fetching viewer showcase ECGs.');
    }
  };

  const handlePredictionFileChange = (event) => {
    setSelectedPredictionFiles(Array.from(event.target.files));
  };

  const handleViewerFileChange = (event) => {
    setSelectedViewerFiles(Array.from(event.target.files));
  };

  const handleUpload = async (folderType) => {
    let filesToUpload = [];
    let uploadEndpoint = '';

    if (folderType === 'prediction') {
      filesToUpload = selectedPredictionFiles;
      uploadEndpoint = 'http://localhost:8000/admin/upload-prediction-ecg';
    } else if (folderType === 'viewer') {
      filesToUpload = selectedViewerFiles;
      uploadEndpoint = 'http://localhost:8000/admin/upload-viewer-ecg';
    }

    if (filesToUpload.length === 0) {
      setMessage(`Please select files to upload for ${folderType} showcase.`);
      return;
    }

    const formData = new FormData();
    filesToUpload.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await axios.post(uploadEndpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMessage(`Uploaded to ${folderType} showcase: ${response.data.filenames.join(', ')}`);
      if (folderType === 'prediction') {
        setSelectedPredictionFiles([]);
        fetchPredictionShowcaseECGs();
      } else if (folderType === 'viewer') {
        setSelectedViewerFiles([]);
        fetchViewerShowcaseECGs();
      }
    } catch (error) {
      console.error(`Error uploading files to ${folderType} showcase:`, error);
      setMessage(`Error uploading files to ${folderType} showcase: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleDelete = async (filename, folderType) => {
    if (window.confirm(`Are you sure you want to delete ${filename} from ${folderType} showcase?`)) {
      try {
        await axios.delete(`http://localhost:8000/admin/delete-showcase-ecg/${filename}?folder_type=${folderType}`);
        setMessage(`${filename} deleted successfully from ${folderType} showcase.`);
        if (folderType === 'prediction') {
          fetchPredictionShowcaseECGs();
        } else if (folderType === 'viewer') {
          fetchViewerShowcaseECGs();
        }
      } catch (error) {
        console.error(`Error deleting ${filename} from ${folderType} showcase:`, error);
        setMessage(`Error deleting ${filename} from ${folderType} showcase: ${error.response?.data?.detail || error.message}`);
      }
    }
  };

const handleRename = async (oldFilename, folderType) => {
  const newFilename = prompt(`Enter new name for ${oldFilename}:`);
  if (newFilename && newFilename.trim() !== '') {
    try {
      // Append folderType as query param
      await axios.put(`http://localhost:8000/admin/rename-showcase-ecg/${oldFilename}?folder_type=${folderType}`, {
        new_filename: newFilename
      });

      setMessage(`${oldFilename} renamed to ${newFilename} successfully.`);

      if (folderType === 'prediction') {
        fetchPredictionShowcaseECGs();
      } else if (folderType === 'viewer') {
        fetchViewerShowcaseECGs();
      }
    } catch (error) {
      console.error(`Error renaming ${oldFilename}:`, error);
      setMessage(`Error renaming ${oldFilename}: ${error.response?.data?.detail || error.message}`);
    }
  }
};


  return (
    <div className="container mx-auto p-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      <h1 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">Admin Panel</h1>

      {message && (
        <div className="mb-4 p-3 rounded-md bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
          {message}
        </div>
      )}

      {/* Prediction Showcase Upload Section */}
      <div className="mb-8 p-6 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-gray-100">Upload Prediction Showcase ECGs</h2>
        <input
          type="file"
          multiple
          onChange={handlePredictionFileChange}
          className="block w-full text-sm text-gray-500 dark:text-gray-400
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100 dark:file:bg-blue-900 dark:file:text-blue-300 dark:hover:file:bg-blue-800"
        />
        <button
          onClick={() => handleUpload('prediction')}
          className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
        >
          <Upload size={20} />
          <span>Upload Prediction Files</span>
        </button>
        {selectedPredictionFiles.length > 0 && (
          <div className="mt-2 text-sm text-gray-600 dark:text-gray-300">
            Selected: {selectedPredictionFiles.map(file => file.name).join(', ')}
          </div>
        )}
      </div>

      {/* Viewer Showcase Upload Section */}
      <div className="mb-8 p-6 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-gray-100">Upload Viewer Showcase ECGs</h2>
        <input
          type="file"
          multiple
          onChange={handleViewerFileChange}
          className="block w-full text-sm text-gray-500 dark:text-gray-400
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100 dark:file:bg-blue-900 dark:file:text-blue-300 dark:hover:file:bg-blue-800"
        />
        <button
          onClick={() => handleUpload('viewer')}
          className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
        >
          <Upload size={20} />
          <span>Upload Viewer Files</span>
        </button>
        {selectedViewerFiles.length > 0 && (
          <div className="mt-2 text-sm text-gray-600 dark:text-gray-300">
            Selected: {selectedViewerFiles.map(file => file.name).join(', ')}
          </div>
        )}
      </div>

      {/* Existing Prediction Showcase ECGs Section */}
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-gray-100">Existing Prediction Showcase ECGs</h2>
        {predictionShowcaseECGs.length === 0 ? (
          <p className="text-gray-600 dark:text-gray-400">No prediction showcase ECGs found.</p>
        ) : (
          <ul className="space-y-3">
            {predictionShowcaseECGs.map((ecg) => (
              <li key={ecg.filename} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg shadow-sm">
                <span className="text-gray-800 dark:text-gray-200 font-medium">{ecg.filename}</span>
                <div>
                  <button
                    onClick={() => handleRename(ecg.filename, 'prediction')}
                    className="p-2 bg-yellow-500 text-white rounded-full hover:bg-yellow-600 transition-colors"
                  >
                    <Pencil size={18} />
                  </button>
                  <button
                    onClick={() => handleDelete(ecg.filename, 'prediction')}
                    className="p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors ml-2"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Existing Viewer Showcase ECGs Section */}
      <div>
        <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-gray-100">Existing Viewer Showcase ECGs</h2>
        {viewerShowcaseECGs.length === 0 ? (
          <p className="text-gray-600 dark:text-gray-400">No viewer showcase ECGs found.</p>
        ) : (
          <ul className="space-y-3">
            {viewerShowcaseECGs.map((ecg) => (
              <li key={ecg.filename} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg shadow-sm">
                <span className="text-gray-800 dark:text-gray-200 font-medium">{ecg.filename}</span>
                <div>
                  <button
                    onClick={() => handleRename(ecg.filename, 'viewer')}
                    className="p-2 bg-yellow-500 text-white rounded-full hover:bg-yellow-600 transition-colors"
                  >
                    <Pencil size={18} />
                  </button>
                  <button
                    onClick={() => handleDelete(ecg.filename, 'viewer')}
                    className="p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors ml-2"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default AdminPanel;
