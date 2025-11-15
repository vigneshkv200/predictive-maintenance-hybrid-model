# 🚀 Predictive Maintenance – Hybrid RUL Model
### LSTM • Autoencoder • Hybrid Meta-Fusion • Streamlit Deployment

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

# 🧭 1. Project Overview

**Predictive Maintenance – Hybrid RUL Model** is an end-to-end AI system for forecasting **Remaining Useful Life (RUL)** of turbofan engines using a combination of:

- **LSTM** → RUL regression  
- **Autoencoder** → anomaly detection  
- **Hybrid Meta-Model** → fuses RUL + anomaly score to deliver stable final RUL  

Supports **NASA CMAPSS FD001**, custom sensor logs, and messy real-world CSV/TXT files.  
Includes a **Streamlit UI** for real-time predictions.

---

# 🏗 2. Architecture Diagram

```
                 ┌──────────────────────────┐
                 │   Raw Sensor Input (24)  │
                 └──────────────┬───────────┘
                                │
                        ┌───────▼────────┐
                        │  Preprocessing │
                        │  - Clean CSV   │
                        │  - Fix TXT     │
                        │  - Scale Input │
                        └───────┬────────┘
                                │
                     ┌──────────▼──────────┐
                     │   LSTM Model        │
                     │ (RUL Regression)    │
                     └──────────┬──────────┘
                                │(pred_rul)
                     ┌──────────▼──────────┐
                     │ Autoencoder Model   │
                     │ (Reconstruction Err)│
                     └──────────┬──────────┘
                                │(anom_score)
                   ┌────────────▼────────────┐
                   │   Hybrid Fusion Model    │
                   │(pred_rul + anomaly → RUL)│
                   └────────────┬────────────┘
                                │
                        ┌───────▼────────┐
                        │  Final RUL     │
                        └────────────────┘
```

---

# 🔄 3. Model Pipeline (Step-by-Step)

### **Step 1 — Data Cleaning**
- Handles CSV/TXT  
- Detects missing headers  
- Removes BOM (`ï»¿`)  
- Fixes inconsistent spacing  

### **Step 2 — Scaling**
- Uses MinMaxScaler (`scaler.pkl`)  

### **Step 3 — Sequence Creation**
- LSTM uses **30 timesteps**  

### **Step 4 — LSTM RUL Prediction**
- Single-value RUL regression  

### **Step 5 — Autoencoder Anomaly Score**
- Reconstruction error  
- Threshold in `threshold.txt`

### **Step 6 — Hybrid Model**
- Inputs: `[lstm_rul, anomaly_score]`  
- Outputs **Final RUL**

### **Step 7 — Streamlit Dashboard**
- Upload → Predict → Visualize  

---

# 🌟 4. Features

✔ NASA CMAPSS FD001 support  
✔ 24 sensor inputs  
✔ LSTM (30 timesteps)  
✔ Autoencoder anomaly detection  
✔ Hybrid meta-learning fusion  
✔ Clean handling of CSV/TXT  
✔ Streamlit UI with charts & metrics  

---

# 📁 5. Folder Structure

```
predictive_maintenance_hybrid_model/
│
├── predictive_maintenance_app/
│   ├── app.py
│   ├── lstm_rul_model.keras
│   ├── hybrid_rul_model.keras
│   ├── autoencoder_model.keras
│   ├── scaler.pkl
│   ├── threshold.txt
│
├── model_testing/
│   └── Model_Testing.ipynb
│
├── train_FD001/...
├── test_FD001/...
├── test_FD002/...
├── test_FD003/...
├── test_FD004/...
│
└── README.md
```

---

# ⚙️ 6. Installation

```bash
pip install streamlit pandas numpy scikit-learn tensorflow matplotlib
```

---

# ▶️ 7. Run the Streamlit App

```bash
cd predictive_maintenance_hybrid_model/predictive_maintenance_app
streamlit run app.py
```

---

# 🖼 8. Screenshots (Placeholders)

```
[Insert Dashboard Screenshot Here]
```

```
[Insert Model Output Screenshot Here]
```

---

# 📉 9. Sample Predictions

```
Input: 30 timesteps of sensor data

LSTM RUL            → 87.32  
Anomaly Score       → 0.0041  
Is Anomaly          → False  
Hybrid Final RUL    → 92.10  
```

---

# 🚀 10. Future Improvements

- FD002–FD004 Hybrid Training  
- FastAPI backend  
- Docker deployment  
- IoT streaming pipeline  
- Cloud inference (AWS/GCP)  

---

# 🛠 11. Tech Stack Badges

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Numpy](https://img.shields.io/badge/Numpy-Arrays-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data-purple?logo=pandas)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn)

---

# 📄 12. License

MIT License

---

# 👨‍💻 13. Author

**Vignesh KV**  
AI/ML Engineer – Final Year  
Bangalore, India  
Deep Learning • Predictive Maintenance • Deployment
