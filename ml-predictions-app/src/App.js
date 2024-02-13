import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handlePrediction = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
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
      }
    } else {
      setError('No file selected.');
      setPredictions([]);
    }
  };

  return (
    <div>
      <h2>
        Priority Pred 
        <small class="text-body-secondary">iction Module</small>
      </h2>
        <input type="file" className="form-control form-control-sm" id="inputGroupFile02" onChange={handleFileUpload} />
        <button type="button" onClick ={handlePrediction} class="btn btn-success">Predict Result</button>
    
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {predictions.length > 0 && (
        <div>
          <h2>Predictions:</h2>
          <ul>
            {predictions.map((prediction, index) => (
              <li key={index}>{prediction}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
