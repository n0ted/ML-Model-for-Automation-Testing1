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
    <div className="container p-3 mb-2 bg-info text-dark" style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center',
      justifyContent: 'start',
      minHeight: '100vh', 
    }}>
      <h1 className="h1name" style={{ 
        marginTop: '20px',
        alignSelf: 'center'
      }}>
        Priority Prediction Module
      </h1>
      <div style={{ 
        marginTop: '20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
      }}>
        <input type="file" className="form-control form-control-sm custom-file-input" id="inputGroupFile02" onChange={handleFileUpload} />
        <div style={{ alignItems: 'center', justifyContent: 'center'}}> {/* Flex container for button and spinner */}
        <button class="button-86" role="button" onClick={handlePrediction}>PREDICT</button>
  {/* <button type="button" className="btn btn-success btn1" style={{ 
    width: '358px', 
    borderRadius: '20px', 
    marginTop: '10px',
    fontWeight: 'bolder',
    marginLeft: '70px',
    marginRight: '20px', // Keep space for spinner

  }} onClick={handlePrediction}>Predict</button> */}
  {/* Placeholder for spinner */}
      <div style={{ 
        width: '50px', // Approximate width of spinner
        height: '50px', // Approximate height of spinner
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        visibility: loading ? 'visible' : 'hidden', // Only show spinner when loading
      }}>
        <div className="spinner-grow" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    </div>
      </div>
      {error && <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>}
      {predictions.length > 0 && (
        <div className="scrollable-container" style={{ 
          maxHeight: '310px', 
          overflowY: 'auto', 
          marginTop: '-28px',  
          borderRadius: '15px', 
          padding: '15px',
        }}>
          <h1 className='pred' style={{ color: '#495057', fontFamily: 'Arial, "Helvetica Neue", Helvetica, sans-serif', fontWeight: 'bold' }}>Predictions</h1>
          <ul style={{ listStyleType: 'none', paddingLeft: '0', color: 'white', padding: '0 20px' }}>
            {predictions.map((prediction, index) => (
              <li key={index} className= "test" style={{ 
                margin: '10px 0', 
                padding: '5px', 
                borderRadius: '10px', 
                background: 'linear-gradient(to left, #009432, #C4E538)',
                boxShadow: '0 2px 4px rgba(0,0,0,1.1)',
                backdropFilter: 'blur(5px)',
                alignContent: 'center',
              }}>
                TC{index + 1}: {prediction}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
