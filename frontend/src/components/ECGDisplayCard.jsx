
import React from 'react';
import Plot from 'react-plotly.js';

const ECGDisplayCard = ({ ecg }) => {
    if (!ecg) {
        return null;
    }

    const { filename, signal, fs, file_size, last_modified, file_type } = ecg;

    const time = Array.from({ length: signal.length }, (_, i) => i / fs);

    return (
        <div className="border p-4 rounded-lg">
            <h3 className="font-semibold text-lg mb-2">{filename}</h3>
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
            <div className="mt-2 text-sm text-gray-600">
                <p>File Size: {file_size} bytes</p>
                <p>Last Modified: {new Date(last_modified * 1000).toLocaleString()}</p>
                <p>File Type: {file_type}</p>
            </div>
        </div>
    );
};

export default ECGDisplayCard;
