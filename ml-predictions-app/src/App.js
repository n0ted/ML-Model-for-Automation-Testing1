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
    <div className="container p-3 mb-2 bg-info text-dark" style={{ minHeight: '100vh', background: 'linear-gradient(to right, #0f2027, #203a43, #2c5364)' }}>
      <h2 style={{ marginRight: '50px', marginTop: '10px' }}>
        Priority Prediction Module
      </h2>
      <input type="file" className="form-control form-control-sm custom-file-input" id="inputGroupFile02" onChange={handleFileUpload} />
      <button type="button" className="btn btn-success" style={{ width: '80px', borderRadius: '15px', marginTop: '10px', marginRight:'10px' }} onClick={handlePrediction}>Predict Result</button>

      {loading && <div className="spinner-border text-primary" role="status"><span className="visually-hidden">Loading...</span></div>}
      {error && <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>}
      {predictions.length > 0 && (
        <div className="scrollable-container" style={{ 
          maxHeight: '500px', 
          overflowY: 'scroll', 
          marginTop: '20px', 
          backgroundColor: 'rgba(237, 247, 255, 0.1)', 
          borderRadius: '15px', 
          border: '0px solid #dee2e6', 
          padding: '15px'
        }}>
          <h2 style={{ color: '#495057' }}>Predictions:</h2>
          <ul style={{ listStyleType: 'none', paddingLeft: '0', color: 'white' }}>
            {predictions.map((prediction, index) => (
              <li key={index} style={{ 
                background: 'rgba(255, 255, 255, 0.6)', 
                margin: '10px 0', 
                padding: '10px', 
                borderRadius: '10px', 
                background: 'linear-gradient(to right, #11998e, #38ef7d)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                backdropFilter: 'blur(5px)'
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
// function App() {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [predictions, setPredictions] = useState([]);
//   const [error, setError] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleFileUpload = (event) => {
//     const file = event.target.files[0];
//     setSelectedFile(file);
//   };

//   const handlePrediction = async () => {
//     if (selectedFile) {
//       const formData = new FormData();
//       formData.append('file', selectedFile);

//       try {
//         setLoading(true);
//         const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
//           headers: {
//             'Content-Type': 'multipart/form-data'
//           }
//         });
//         setPredictions(response.data.predictions);
//         setError(null);
//       } catch (error) {
//         setError('Error predicting: ' + error.message);
//         setPredictions([]);
//       } finally {
//         setLoading(false);
//       }
//     } else {
//       setError('No file selected.');
//       setPredictions([]);
//     }
//   };

//   return (
//     <div className="container p-3 mb-2 bg-info text-dark" style={{ minHeight: '100vh', background: 'linear-gradient(to right, #0f2027, #203a43, #2c5364)' }}>
//       <h2 style={{ marginRight: '50px', marginTop: '10px' }}>
//         Priority Prediction Module
//       </h2>
//       <input type="file" className="form-control form-control-sm custom-file-input" id="inputGroupFile02" onChange={handleFileUpload} />
//       <button type="button" onClick={handlePrediction} className="btn btn-success" style={{ width: '80px', borderRadius: '39px', marginTop: '10px' }}>Predict Result</button>

//       {loading && <div className="spinner-border text-primary" role="status"><span className="visually-hidden">Loading...</span></div>}
//       {error && <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>}
//       {predictions.length > 0 && (
//         <div style={{ maxHeight: '500px', overflowY: 'scroll', marginTop: '20px', background: 'linear-gradient(to bottom, #8e9eab, #eef2f3)', borderRadius: '15px', border: '2px solid #dee2e6', padding: '15px' }}>
//           <h2 style={{ color: '#495057' }}>Predictions:</h2>
//           <ul style={{ listStyleType: 'none', paddingLeft: '0', color: '#343a40' }}>
//             {predictions.map((prediction, index) => (
//               <li key={index} style={{ background: 'linear-gradient(to right, #11998e, #38ef7d)', margin: '10px 0', padding: '10px', borderRadius: '10px', boxShadow: '0 2px 4px rgba(0,0,0,.1)' }}>
//                 TC{index + 1}: {prediction}
//               </li>
//             ))}
//           </ul>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;
