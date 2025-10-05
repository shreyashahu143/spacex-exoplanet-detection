// Wait for the entire HTML document to be loaded before running the script
document.addEventListener('DOMContentLoaded', function() {
    
    // Get references to all the interactive HTML elements
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const loader = document.getElementById('loader');
    const validationContainer = document.getElementById('validation-container');
    const validateButton = document.getElementById('validate-button');

    // This variable will hold the crucial data passed from the first step to the second
    let storedPredictionData = null;

    // --- WORKFLOW STEP 1: Initial XGBoost Prediction ---
    form.addEventListener('submit', function(event) {
        // Prevent the default form action (which would reload the page)
        event.preventDefault();

        // Reset the UI for a new prediction
        loader.style.display = 'block';
        resultContainer.style.display = 'none';
        validationContainer.style.display = 'none';
        resultText.textContent = '';
        storedPredictionData = null; // Clear previous results

        // Gather the 6 input features from the form
        const formData = {
            'koi_period': parseFloat(document.getElementById('koi_period').value),
            'koi_depth': parseFloat(document.getElementById('koi_depth').value),
            'koi_duration': parseFloat(document.getElementById('koi_duration').value),
            'koi_srad': parseFloat(document.getElementById('koi_srad').value),
            'koi_impact': parseFloat(document.getElementById('koi_impact').value),
            'koi_steff': parseFloat(document.getElementById('koi_steff').value)
        };
        
        // --- FETCH REQUEST 1: Get Initial Prediction ---
        fetch('/initial_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData) 
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Initial prediction failed.'); });
            }
            return response.json();
        })
        .then(data => {
            // --- On Success ---
            // 1. Store the response data for the next step
            storedPredictionData = data; // Contains xgb_probability and processed_data

            // 2. Display the initial XGBoost result to the user
            const xgbProbability = (data.xgb_probability * 100).toFixed(1);
            resultText.innerHTML = `<strong>Initial Finding (XGBoost Model):</strong><br> There is a 
                <span class="highlight">${xgbProbability}%</span> probability this is a viable candidate.`;
            
            // 3. Show the result container and the "Validate" button
            resultContainer.style.display = 'block';
            validationContainer.style.display = 'block';
        })
        .catch(error => {
            resultContainer.style.display = 'block';
            resultText.textContent = `Error: ${error.message}`;
            console.error('Initial predict error:', error);
        })
        .finally(() => {
            loader.style.display = 'none'; // Always hide the loader
        });
    });

    // --- WORKFLOW STEP 2: Full Validation and Redirect ---
    validateButton.addEventListener('click', function() {
        if (!storedPredictionData) {
            alert("An error occurred. Please run an initial prediction first.");
            return;
        }

        // Update UI to show the validation process has started
        loader.style.display = 'block';
        validateButton.disabled = true;
        validateButton.innerText = "Running Full Analysis...";

        // --- FETCH REQUEST 2: Get Final Validation ---
        // Prepare the payload using the data we stored from the first request
        const validationPayload = {
            scaled_features: storedPredictionData.processed_data.scaled,
            imputed_features: storedPredictionData.processed_data.imputed,
            xgb_probability: storedPredictionData.xgb_probability
        };

        fetch('/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(validationPayload)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Validation request failed.'); });
            }
            return response.json();
        })
        .then(finalResults => {
            // --- On Success ---
            // 1. Store the complete results object in the browser's sessionStorage.
            // This makes the data available to the dashboard page.
            sessionStorage.setItem('analysisResults', JSON.stringify(finalResults));
            
            // 2. Redirect the user to the new dashboard page to view the results.
            window.location.href = '/dashboard';
        })
        .catch(error => {
            resultText.textContent = `Error: ${error.message}`;
            console.error('Validation process error:', error);
            loader.style.display = 'none';
            validateButton.disabled = false;
            validateButton.innerText = "Validate with CNN & Ensemble";
        });
    });
});

