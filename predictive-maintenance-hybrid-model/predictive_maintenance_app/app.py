import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
from tensorflow.keras.models import load_model
from datetime import datetime

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# --------------------------------------------
# File Names
# --------------------------------------------
LSTM_MODEL = "lstm_rul_model.keras"
AUTOENC_MODEL = "autoencoder_model.keras"
HYBRID_MODEL = "hybrid_rul_model.keras"
SCALER_FILE = "scaler.pkl"
THRESHOLD_FILE = "threshold.txt"

# Sensor names (24 features)
sensor_columns = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SEQ_LEN = 30
DEFAULT_THRESHOLD = 0.0040565341624282


# --------------------------------------------
# LOAD MODELS + SCALER + THRESHOLD
# --------------------------------------------
def load_all():
    if not os.path.exists(SCALER_FILE):
        raise FileNotFoundError("Scaler file missing.")

    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    lstm = load_model(LSTM_MODEL)
    autoenc = load_model(AUTOENC_MODEL)
    hybrid = load_model(HYBRID_MODEL)

    if os.path.exists(THRESHOLD_FILE):
        threshold = float(open(THRESHOLD_FILE).read().strip())
    else:
        threshold = DEFAULT_THRESHOLD

    return scaler, lstm, autoenc, hybrid, threshold


# --------------------------------------------
# CLEAN UPLOADED FILE – *THE MOST IMPORTANT FUNCTION*
# --------------------------------------------
def clean_uploaded_file(file):
    raw = file.read()

    # HANDLE BOM (ï»¿)
    try:
        text = raw.decode("utf-8-sig")
    except:
        text = raw.decode(errors="ignore")

    # Try CSV first
    try:
        df = pd.read_csv(io.StringIO(text), header=None)
    except:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)

    # If there is ONLY 1 column -> split manually
    if df.shape[1] == 1:
        df = df[0].str.split(r"\s+|,|;", expand=True)

    # Drop WhatsApp/Excel index columns like: 0,1,2,3
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(axis=1, how="all")

    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop all-nan cols again
    df = df.dropna(axis=1, how="all")

    # FIX for your common error: 5x25 case → pick last 24 VALID columns
    if df.shape[1] == 25:
        df = df.iloc[:, -24:]
        df.columns = sensor_columns
        return df

    # Normal NASA shapes
    if df.shape[1] == 24:
        df.columns = sensor_columns
    elif df.shape[1] == 26:
        df.columns = ["unit", "time"] + sensor_columns
    elif df.shape[1] == 27:
        df = df.iloc[:, :27]
        df.columns = ["unit", "time"] + sensor_columns + ["RUL"]
    else:
        # last chance rescue: pick last 24 numeric cols
        if df.shape[1] > 24:
            df = df.iloc[:, -24:]
            df.columns = sensor_columns
        else:
            raise ValueError(f"Unsupported file shape: {df.shape}")

    return df


# --------------------------------------------
# BUILD 30-ROW SEQUENCE FOR LSTM
# --------------------------------------------
def build_sequence(df_scaled):
    X = df_scaled[sensor_columns].values
    n = X.shape[0]

    if n >= SEQ_LEN:
        seq = X[-SEQ_LEN:]
    else:
        # pad by repeating last row
        last = X[-1:]
        seq = np.vstack([X, np.repeat(last, SEQ_LEN - n, axis=0)])

    return seq.reshape(1, SEQ_LEN, 24)


# --------------------------------------------
# RUN PREDICTION
# --------------------------------------------
def run_prediction(df_clean, scaler, lstm, autoenc, hybrid, threshold):
    sensor_vals = df_clean[sensor_columns].values

    scaled = scaler.transform(sensor_vals)
    seq = build_sequence(pd.DataFrame(scaled, columns=sensor_columns))

    # LSTM
    lstm_pred = float(lstm.predict(seq, verbose=0)[0][0])

    # Autoencoder anomaly
    last_row = scaled[-1:].astype(float)
    recon = autoenc.predict(last_row, verbose=0)
    anomaly_score = float(np.mean((last_row - recon) ** 2))
    is_anomaly = anomaly_score > threshold

    # Hybrid
    hybrid_in = np.array([[anomaly_score, int(is_anomaly), lstm_pred]])
    hybrid_pred = float(hybrid.predict(hybrid_in, verbose=0)[0][0])

    return lstm_pred, anomaly_score, is_anomaly, hybrid_pred


# --------------------------------------------
# STREAMLIT UI
# --------------------------------------------
st.title("🚀 Predictive Maintenance – Hybrid AI Model")

try:
    scaler, lstm, autoenc, hybrid, threshold = load_all()
    st.success("Models loaded successfully.")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

file = st.file_uploader("📁 Upload sensor CSV / NASA TXT", type=["txt", "csv"])

if file:
    try:
        df_clean = clean_uploaded_file(file)
        st.success(f"File OK: {df_clean.shape}")
        st.dataframe(df_clean.head())

        if st.button("Run Prediction"):
            lstm_pred, anomaly_score, is_anomaly, hybrid_pred = run_prediction(
                df_clean, scaler, lstm, autoenc, hybrid, threshold
            )

            col1, col2 = st.columns(2)
            col1.metric("🔧 LSTM RUL", f"{lstm_pred:.2f}")
            col2.metric("🚨 Hybrid RUL", f"{hybrid_pred:.2f}")

            st.metric("⚠ Anomaly Score", f"{anomaly_score:.4f}")
            st.metric("Is Anomaly?", str(is_anomaly))

            st.line_chart(df_clean[sensor_columns].tail(10))

    except Exception as e:
        st.error(f"Error processing: {e}")

st.write("---")
st.caption("Built by *Vignesh KV* – AIML Engineer ")