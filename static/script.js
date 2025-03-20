const API_BASE_URL = "https://speech-emotion-detection-1.onrender.com";

function getPrediction() {
    fetch(${API_BASE_URL}/predict)
        .then(response => response.json())
        .then(data => alert("Prediction: " + data.message))
        .catch(error => console.error("Error:", error));
}