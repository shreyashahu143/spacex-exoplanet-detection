import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app) # Enable CORS

# --- File Configuration ---
MODEL_FILE_NAME = 'best_exoplanet_model_final.pkl'
COLUMNS_FILE_NAME = 'model_column.pkl'
SCALER_FILE_NAME = 'scaler.pkl' # The crucial missing piece

print("--- Initializing Server ---")
try:
    model = joblib.load(MODEL_FILE_NAME)
    MODEL_COLUMNS = joblib.load(COLUMNS_FILE_NAME)
    scaler = joblib.load(SCALER_FILE_NAME)
    print("✅ Model, column order, and feature scaler loaded successfully!")
except FileNotFoundError:
    print(f"🔴 FATAL ERROR: Could not find '{MODEL_FILE_NAME}', '{COLUMNS_FILE_NAME}', or '{SCALER_FILE_NAME}'.")
    print("🔴 ACTION REQUIRED: Ensure you have saved your trained StandardScaler object as 'scaler.pkl'.")
    model, MODEL_COLUMNS, scaler = None, None, None
except Exception as e:
    print(f"🔴 FATAL ERROR: An unexpected error occurred while loading files: {e}")
    model, MODEL_COLUMNS, scaler = None, None, None


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with fundamental exoplanet attributes and engineers the
    full set of features required by the model, matching the neww.csv structure.
    """
    df_engineered = df.copy()

    # --- 1. Physical Constants ---
    G = 6.67430e-11  # Gravitational Constant (m^3 kg^-1 s^-2)
    R_SUN = 6.957e8   # Solar Radius (m)
    M_SUN = 1.989e30  # Solar Mass (kg)
    R_SUN_TO_R_EARTH = R_SUN / 6.371e6 # ~109.2

    # --- 2. Core Feature Calculations (names matched to neww.csv) ---
    df_engineered['koi_ror'] = np.sqrt(df_engineered['koi_depth'] / 1_000_000)
    df_engineered['koi_prad'] = df_engineered['koi_ror'] * df_engineered['koi_srad'] * R_SUN_TO_R_EARTH
    p_hours = df_engineered['koi_period'] * 24
    term_inside_sqrt = (1 + df_engineered['koi_ror'])**2 - df_engineered['koi_impact']**2
    term_inside_sqrt[term_inside_sqrt < 0] = 0
    koi_dor = (p_hours / (df_engineered['koi_duration'] * np.pi)) * np.sqrt(term_inside_sqrt)
    p_seconds = df_engineered['koi_period'] * 24 * 3600
    df_engineered['koi_srho'] = (3 * np.pi / (G * p_seconds**2)) * koi_dor**3
    r_star_meters = df_engineered['koi_srad'] * R_SUN
    df_engineered['koi_smass'] = (df_engineered['koi_srho'] * (4/3) * np.pi * r_star_meters**3) / M_SUN
    m_star_kg = df_engineered['koi_smass'] * M_SUN
    g_ms2 = (G * m_star_kg) / r_star_meters**2
    g_cgs = g_ms2 * 100
    df_engineered['koi_slogg'] = np.log10(g_cgs.replace(0, np.nan))
    a_meters = koi_dor * r_star_meters
    df_engineered['koi_sma'] = a_meters / 1.496e11
    ratio = np.clip(df_engineered['koi_impact'] / koi_dor, -1.0, 1.0)
    df_engineered['koi_incl'] = np.degrees(np.arccos(ratio))
    df_engineered['koi_teq'] = df_engineered['koi_steff'] * np.sqrt(1 / (2 * koi_dor))
    df_engineered['koi_insol'] = ((df_engineered['koi_steff'] / 5778.0)**4) * ((1 / koi_dor)**2)
    delta = df_engineered['koi_depth'] / 1_000_000.0
    term1 = 1470 / df_engineered['koi_period'].replace(0, np.nan)
    term2 = (df_engineered['koi_duration'] / 24) / 0.0204167
    inside_sqrt = (term1 * term2).fillna(0)
    inside_sqrt[inside_sqrt < 0] = 0
    df_engineered['koi_model_snr'] = delta * df_engineered['koi_srad'] * df_engineered['koi_steff'] * np.sqrt(inside_sqrt)
    df_engineered['koi_count'] = df_engineered.get('koi_count', 1)
    df_engineered['koi_num_transits'] = np.floor(1470 / df_engineered['koi_period'].replace(0, np.nan)).astype('Int64')
    df_engineered['koi_smet'] = df_engineered.get('koi_smet', 0.0)
    df_engineered['koi_kepmag'] = df_engineered.get('koi_kepmag', 14.0)

    # --- Final Cleaning ---
    df_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_engineered.fillna(0, inplace=True)
    return df_engineered

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # FIX: Explicitly check if each required object is None to avoid pandas' ambiguous truth value error.
    if model is None or MODEL_COLUMNS is None or scaler is None:
        return jsonify({"error": "Model, Columns, or Scaler not loaded. Check server logs."}), 500

    json_data = request.get_json()
    print(f"\n\n--- New Prediction Request ---")
    print(f"Received observational features: {json_data}")
    
    query_df = pd.DataFrame([json_data])
    for col in query_df.columns:
        query_df[col] = pd.to_numeric(query_df[col], errors='coerce')
    
    # Step 1: Engineer the full set of features
    query_df_engineered = engineer_all_features(query_df)

    # Step 2: Align columns to match the model's training data
    query_df_aligned = query_df_engineered.reindex(columns=MODEL_COLUMNS, fill_value=0)
    query_df_aligned.fillna(0, inplace=True)
    
    print("\n--- Data BEFORE Scaling (Raw Physical Units) ---")
    print(query_df_aligned[['koi_period', 'koi_steff', 'koi_prad', 'koi_model_snr']].head().to_string())

    # Step 3: CRITICAL - Scale the features using the loaded scaler
    scaled_features = scaler.transform(query_df_aligned)
    query_df_scaled = pd.DataFrame(scaled_features, columns=MODEL_COLUMNS)

    print("\n--- Data AFTER Scaling (Model-Ready Units) ---")
    print(query_df_scaled[['koi_period', 'koi_steff', 'koi_prad', 'koi_model_snr']].head().to_string())
    print("--------------------------------------------------\n")

    try:
        # Make the prediction using the SCALED data
        prediction = model.predict(query_df_scaled)
        prediction_proba = model.predict_proba(query_df_scaled)
        result_text = 'Likely a Planet' if prediction[0] == 1 else 'Likely Not a Planet'
        confidence = f"{prediction_proba[0][prediction[0]] * 100:.2f}%"
        
        print(f"Prediction: {result_text} with {confidence} confidence.")
        return jsonify({"prediction": result_text, "confidence": confidence})

    except Exception as e:
        print(f"🔴 ERROR during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction. Details: {e}"}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
