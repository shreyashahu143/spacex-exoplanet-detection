import pickle
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# --- NEW IMPORTS FOR PYTORCH ---
import torch
import torch.nn as nn

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# =============================================================================
# WORKFLOW A: COMPLEX FEATURE ENGINEERING (Scikit-learn XGBoost Model)
# =============================================================================

print("--- Loading Artifacts for Workflow A: Complex Feature Engineering ---")
try:
    # STANDARDIZED FILENAMES: Rename your files to match these names for consistency
    complex_model = joblib.load('best_exoplanet_model_final (3).pkl')
    complex_model_columns = joblib.load('model_column copy.pkl')
    complex_scaler = joblib.load('scaler (5).pkl')
    print("✅ [Workflow A] Models and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"🔴 WARNING [Workflow A]: Could not load model files. The '/predict_engineered' route will not work. Details: {e}")
    complex_model, complex_model_columns, complex_scaler = None, None, None

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with fundamental exoplanet attributes and engineers the
    full set of features required by the complex XGBoost model.
    """
    # This entire function is preserved as you requested.
    df_engineered = df.copy()
    # --- Physical Constants & Calculations ---
    G = 6.67430e-11
    R_SUN = 6.957e8
    M_SUN = 1.989e30
    R_SUN_TO_R_EARTH = R_SUN / 6.371e6
    df_engineered['koi_ror'] = np.sqrt(df_engineered['koi_depth'] / 1_000_000)
    df_engineered['koi_prad'] = df_engineered['koi_ror'] * df_engineered['koi_srad'] * R_SUN_TO_R_EARTH
    p_hours = df_engineered['koi_period'] * 24
    # CORRECTED: Use **2 for squaring, not *2
    term_inside_sqrt = (1 + df_engineered['koi_ror'])**2 - df_engineered['koi_impact']**2
    term_inside_sqrt[term_inside_sqrt < 0] = 0
    koi_dor = (p_hours / (df_engineered['koi_duration'] * np.pi)) * np.sqrt(term_inside_sqrt)
    p_seconds = df_engineered['koi_period'] * 24 * 3600
    # CORRECTED: Use **2 for squaring and **3 for cubing
    df_engineered['koi_srho'] = (3 * np.pi / (G * p_seconds**2)) * koi_dor**3 if G > 0 and not p_seconds.eq(0).any() else 0
    r_star_meters = df_engineered['koi_srad'] * R_SUN
    df_engineered['koi_smass'] = (df_engineered['koi_srho'] * (4/3) * np.pi * r_star_meters**3) / M_SUN
    m_star_kg = df_engineered['koi_smass'] * M_SUN
    g_ms2 = (G * m_star_kg) / r_star_meters**2
    g_cgs = g_ms2 * 100
    df_engineered['koi_slogg'] = np.log10(g_cgs.replace(0, np.nan))
    a_meters = koi_dor * r_star_meters
    df_engineered['koi_sma'] = a_meters / 1.496e11
    ratio = np.clip(df_engineered['koi_impact'] / koi_dor.replace(0, 1e-9), -1.0, 1.0)
    df_engineered['koi_incl'] = np.degrees(np.arccos(ratio))
    df_engineered['koi_teq'] = df_engineered['koi_steff'] * np.sqrt(1 / (2 * koi_dor.replace(0, 1e-9)))
    # CORRECTED: Use **4 for power of 4 and **2 for squaring
    df_engineered['koi_insol'] = ((df_engineered['koi_steff'] / 5778.0)**4) * ((1 / koi_dor.replace(0, 1e-9))**2)
    # --- Final Cleaning ---
    df_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_engineered.fillna(0, inplace=True)
    return df_engineered

@app.route('/predict_engineered', methods=['POST'])
def predict_engineered():
    """Endpoint for the complex feature engineering workflow."""
    if not all([complex_model, complex_model_columns, complex_scaler]):
        return jsonify({"error": "[Workflow A] Model files not loaded. Check server logs."}), 500
    
    json_data = request.get_json()
    query_df = pd.DataFrame([json_data])
    query_df_engineered = engineer_all_features(query_df)
    query_df_aligned = query_df_engineered.reindex(columns=complex_model_columns, fill_value=0)
    scaled_features = complex_scaler.transform(query_df_aligned)
    
    prediction_proba = complex_model.predict_proba(scaled_features)
    planet_probability = prediction_proba[0][1]
    
    status = "Confirmed Planet" if planet_probability >= 0.5 else "Not a Planet"
    return jsonify({'status': status, 'planet_probability': float(planet_probability)})


# =============================================================================
# WORKFLOW B: STACKED ENSEMBLE (PyTorch & Scikit-learn Models)
# =============================================================================

# --- PYTORCH MODEL DEFINITION ---
class PyTorchCNN1D(nn.Module):
    """Defines the 1D CNN architecture using PyTorch."""
    def __init__(self):
        super(PyTorchCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        self.flattened_size = 64 * 22 
        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout2(torch.relu(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, self.flattened_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) # Output raw logits
        return x

# --- Loading Artifacts for Workflow B ---
print("\n--- Loading Artifacts for Workflow B: Stacked Ensemble ---")
XGB_FEATURES_STACKED = ['koi_period', 'koi_depth', 'koi_duration', 'koi_srad', 'koi_impact', 'koi_steff']
N_FEATURES_XGB_MODEL_STACKED = 27
N_PADDING_ZEROS_STACKED = N_FEATURES_XGB_MODEL_STACKED - len(XGB_FEATURES_STACKED)
CURVE_LENGTH = 100
DEVICE = torch.device("cpu")

try:
    # STANDARDIZED FILENAMES: Rename your files to match these names
    with open('best_exoplanet_model_final.pkl', 'rb') as f:
        xgb_model_stacked = pickle.load(f)
    
    cnn_model_stacked = PyTorchCNN1D().to(DEVICE)
    cnn_model_stacked.load_state_dict(torch.load('cnn_model_pytorch (1).pth', map_location=DEVICE)) 
    cnn_model_stacked.eval()
    
    meta_model_stacked = joblib.load('meta_model_rf (1).pkl')
    
    imputer_stacked = joblib.load('imputer (2).pkl')
    scaler_stacked = joblib.load('scaler (5).pkl')
    print("✅ [Workflow B] All models and preprocessors loaded successfully!")
except FileNotFoundError as e:
    print(f"🔴 WARNING [Workflow B]: Could not load model files. The validation routes will not work. Details: {e}")
    xgb_model_stacked, cnn_model_stacked, meta_model_stacked, imputer_stacked, scaler_stacked = [None]*5

def generate_light_curve(row_data, num_points):
    """Generates a synthetic light curve for the CNN model."""
    data_dict = dict(zip(XGB_FEATURES_STACKED, row_data))
    period_days = data_dict.get('koi_period', 0)
    if period_days <= 0: return np.ones(num_points)
    duration_hr = data_dict.get('koi_duration', 0)
    depth_ppt = data_dict.get('koi_depth', 0)
    fractional_duration = (duration_hr / 24) / period_days
    depth_flux = depth_ppt / 1_000_000.0
    ingress_egress_fraction = fractional_duration * 0.2
    time = np.linspace(0, 1, num_points)
    flux = np.ones(num_points)
    center, half_duration = 0.5, fractional_duration / 2
    t1, t2 = center - half_duration, center - half_duration + ingress_egress_fraction
    t3, t4 = center + half_duration - ingress_egress_fraction, center + half_duration
    flux[(time >= t2) & (time <= t3)] = 1.0 - depth_flux
    if ingress_egress_fraction > 0:
        ingress_slope = depth_flux / ingress_egress_fraction
        flux[(time > t1) & (time < t2)] = 1.0 - ingress_slope * (time[(time > t1) & (time < t2)] - t1)
        flux[(time > t3) & (time < t4)] = (1.0 - depth_flux) + ingress_slope * (time[(time > t3) & (time < t4)] - t3)
    return flux

@app.route('/initial_predict', methods=['POST'])
def initial_predict():
    """Handles the first step of Workflow B: preprocessing and XGBoost prediction."""
    if not all([xgb_model_stacked, imputer_stacked, scaler_stacked]):
        return jsonify({'error': '[Workflow B] Initial models not loaded.'}), 500
    
    input_data = request.get_json()
    
    # Use a DataFrame to preserve feature names and avoid the UserWarning
    input_df = pd.DataFrame([input_data], columns=XGB_FEATURES_STACKED)
    input_imputed = imputer_stacked.transform(input_df)
    input_scaled = scaler_stacked.transform(input_imputed)
    
    zero_padding = np.zeros((1, N_PADDING_ZEROS_STACKED))
    input_compatible = np.hstack((input_scaled, zero_padding))
    
    xgb_proba = xgb_model_stacked.predict_proba(input_compatible)[:, 1][0]
    
    return jsonify({
        "xgb_probability": float(xgb_proba),
        "processed_data": {
            "imputed": input_imputed.tolist()[0], 
            "scaled": input_scaled.tolist()[0]
        }
    })

@app.route('/validate', methods=['POST'])
def validate():
    """Handles the second step of Workflow B."""
    if not all([cnn_model_stacked, meta_model_stacked]):
        return jsonify({'error': '[Workflow B] Validation models not loaded.'}), 500
        
    data = request.get_json()
    input_scaled = np.array([data['scaled_features']])
    input_imputed = np.array(data['imputed_features'])
    xgb_proba = data['xgb_probability']
    
    with torch.no_grad():
        lc_flux = generate_light_curve(input_imputed, CURVE_LENGTH)
        lc_tensor = torch.from_numpy(lc_flux).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits = cnn_model_stacked(lc_tensor)
        # Apply sigmoid to the raw logits to get a probability
        cnn_proba = torch.sigmoid(logits).item()

    meta_features_np = np.column_stack((input_scaled, xgb_proba, cnn_proba))
    final_proba = meta_model_stacked.predict_proba(meta_features_np)[:, 1][0]
    
    time_for_chart = np.linspace(0, 1, CURVE_LENGTH)
    return jsonify({
        "xgb_probability": float(xgb_proba), 
        "cnn_probability": float(cnn_proba),
        "final_prediction": "Exoplanet" if final_proba > 0.5 else "False Positive",
        "final_confidence": float(final_proba),
        "light_curve": {"time": time_for_chart.tolist(), "flux": lc_flux.tolist()}
    })

# =============================================================================
# COMMON ROUTES
# =============================================================================
@app.route('/')
def home():
    """Renders the main homepage."""
    return render_template('homepage.html')

@app.route('/index')
def index():
    """Renders the page for the two-step stacked ensemble workflow."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Renders the analysis dashboard page."""
    return render_template('dashboard.html')

# =============================================================================
# APP RUN
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5001)

