// Wait for the entire HTML document to be loaded before running the script
document.addEventListener('DOMContentLoaded', function() {
    
    // 1. Get data from the browser's sessionStorage
    // This is more robust than URL parameters for complex data.
    const resultsString = sessionStorage.getItem('analysisResults');

    if (resultsString) {
        // If data is found, parse it from a string back into an object
        const data = JSON.parse(resultsString);
        renderDashboard(data);
    } else {
        // Handle the case where the dashboard is opened directly without data
        const dashboardElement = document.querySelector('.dashboard-container');
        if (dashboardElement) {
            dashboardElement.innerHTML = '<p style="color:red; text-align:center;">Error: No prediction data found. Please return to the form and run an analysis first.</p>';
        }
    }

    // --- Function to update and show the Pictorial Dashboard ---
    function renderDashboard(data) {
        // Extract data for easier access
        const finalPrediction = data.final_prediction;
        const finalProba = data.final_confidence;
        const xgbProba = data.xgb_probability;
        const cnnProba = data.cnn_probability;
        const isExoplanet = finalPrediction === 'Exoplanet';
        
        // --- 1. Update Summary Scorecards ---
        const finalPredictionEl = document.getElementById('final-prediction');
        const finalConfidenceEl = document.getElementById('final-confidence');
        const xgbScoreEl = document.getElementById('xgb-score');
        const cnnScoreEl = document.getElementById('cnn-score');

        finalPredictionEl.textContent = finalPrediction;
        finalConfidenceEl.textContent = `${(finalProba * 100).toFixed(1)}%`;
        xgbScoreEl.textContent = `${(xgbProba * 100).toFixed(1)}%`;
        cnnScoreEl.textContent = `${(cnnProba * 100).toFixed(1)}%`;

        // Add color coding based on the final prediction
        if (isExoplanet) {
            finalPredictionEl.classList.add('exoplanet');
            finalConfidenceEl.classList.add('exoplanet');
        } else {
            finalPredictionEl.classList.add('false-positive');
            finalConfidenceEl.classList.add('false-positive');
        }

        // --- 2. Update Description (Interpretation Text) ---
        const descriptionEl = document.getElementById('description-text');
        let description = `The <strong>Stacked Ensemble</strong> model has made a final prediction of <strong>'${finalPrediction}'</strong> with <strong>${(finalProba * 100).toFixed(1)}%</strong> confidence. This result is based on an analysis of both the candidate's physical features and its simulated transit light curve.`;
        
        // Add a special note if the models disagree significantly
        if (Math.abs(xgbProba - cnnProba) > 0.3) { // If probabilities differ by more than 30%
            description += `<br><br><strong>Note:</strong> A significant divergence was detected between the feature-based model (XGBoost) and the light curve model (CNN), suggesting a complex or ambiguous signal. The final result relies on the ensemble model's ability to weigh this conflict.`;
        }
        descriptionEl.innerHTML = description;

        // --- 3. Render the Charts ---
        renderDonutChart(xgbProba, cnnProba);
        renderLightCurveChart(data.light_curve.time, data.light_curve.flux, finalPrediction);
    }

    // --- Function to Create Donut Chart ---
    function renderDonutChart(xgb, cnn) {
        const ctx = document.getElementById('donutChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['XGBoost Probability', 'CNN Probability'],
                datasets: [{
                    data: [xgb, cnn],
                    backgroundColor: ['#3498db', '#f1c40f'],
                    borderColor: '#ffffff',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed !== null) {
                                    label += (context.parsed * 100).toFixed(1) + '%';
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

    // --- Function to Create Light Curve Chart ---
    function renderLightCurveChart(time, flux, prediction) {
        const ctx = document.getElementById('lightCurveChart').getContext('2d');
        const color = (prediction === 'Exoplanet') ? '#27ae60' : '#c0392b';

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: time,
                datasets: [{
                    label: 'Normalized Flux',
                    data: flux,
                    borderColor: color,
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Normalized Orbital Phase (Time)' },
                        ticks: { maxTicksLimit: 10 }
                    },
                    y: {
                        title: { display: true, text: 'Normalized Flux' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
});

