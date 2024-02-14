import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handlePrediction = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        setLoading(true);
        const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        setPredictions(response.data.predictions);
        setError(null);
      } catch (error) {
        setError('Error predicting: ' + error.message);
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    } else {
      setError('No file selected.');
      setPredictions([]);
    }
  };

  return (
    <div className="p-3 mb-2 bg-info text-dark">
      <h2>
        Priority Prediction Module
        <small className="text-muted"> (Prediction Module)</small>
      </h2>
      <input type="file" className="form-control form-control-sm" id="inputGroupFile02" onChange={handleFileUpload} />
      <button type="button" onClick={handlePrediction} className="btn btn-success">Predict Result</button>

      {loading && <div className="spinner-border text-primary" role="status"><span className="visually-hidden">Loading...</span></div>}
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {predictions.length > 0 && (
        <div>
          <h2>Predictions:</h2>
          <ul>
            {predictions.map((prediction, index) => (
              <li key={index}>TC{index + 1}: {prediction}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
