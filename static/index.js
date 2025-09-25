// Wait for the entire HTML document to be loaded before running the script
document.addEventListener('DOMContentLoaded', function() {
    
    // Get references to the HTML elements
    const form = document.getElementById('prediction-form');
    const resultText = document.getElementById('result-text');
    const loader = document.getElementById('loader');

    // Add an event listener for the form's 'submit' event
    form.addEventListener('submit', function(event) {
        // Prevent the default form action (which would reload the page)
        event.preventDefault();

        // Show the loader and clear previous results
        loader.style.display = 'block';
        resultText.innerText = '';

        // --- Step 1: Gather the data from the form ---
        // Create a JavaScript object to hold the form data.
        // The keys MUST match the feature names your model expects.
        const formData = {
            "koi_period": parseFloat(document.getElementById('koi_period').value),
            "koi_duration": parseFloat(document.getElementById('koi_duration').value),
            "koi_depth": parseFloat(document.getElementById('koi_depth').value),
            "koi_prad": parseFloat(document.getElementById('koi_prad').value),
            "koi_teq": parseFloat(document.getElementById('koi_teq').value),
            "koi_srad": parseFloat(document.getElementById('koi_srad').value),
            "koi_impact": parseFloat(document.getElementById('koi_impact').value),
            "koi_steff": parseFloat(document.getElementById('koi_steff').value)
            
            // IMPORTANT: In a real app, you would add ALL other features
            // your model was trained on here, setting them to a default (e.g., 0)
            // or getting them from the user.
        };
        // ADD THIS LINE TO CHECK YOUR WORK
        console.log('Data being sent to backend:', formData);
        // --- Step 2: Send data to the Flask API ---
        // Use the Fetch API to make a POST request.
        fetch('https://spacex-exoplanet-detection.onrender.com/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json' // Tell the server we're sending JSON
            },
            // Convert the JavaScript object to a JSON string for the request body
            body: JSON.stringify(formData) 
        })
        .then(response => response.json()) // Parse the JSON response from the server
        .then(data => {
            // --- Step 3: Display the result ---
            loader.style.display = 'none'; // Hide the loader
            
            // Update the result text with the prediction from the server
            resultText.innerText = data.prediction;
            
            // Change text color based on prediction
            if (data.prediction === 'Planet') {
                resultText.style.color = '#27ae60'; // Green
            } else {
                resultText.style.color = '#c0392b'; // Red
            }
        })
        .catch(error => {
            // --- Step 4: Handle any errors ---
            loader.style.display = 'none'; // Hide the loader
            console.error('Error:', error);
            resultText.innerText = 'Error: Could not connect to the server.';
            resultText.style.color = '#c0392b';
        });
    });
});