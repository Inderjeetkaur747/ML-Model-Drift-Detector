#  ML Monitoring System with Data Drift Detection & Auto-Retraining

##  Overview
This project is an end-to-end Machine Learning system designed to monitor model performance in production by detecting data drift and automatically triggering model retraining.

It simulates a real-world fraud detection pipeline where incoming data distributions may change over time, affecting model accuracy.

---
**Live App:** https: //ml-model-drift-detector-dkfqsnvvgqvndre9hrcbqz.streamlit.app/
**API Docs (Swagger):** http://localhost:8000/docs 

## Problem Statement
Machine learning models degrade over time due to **data drift**. This project solves that problem by:

- Detecting drift in incoming data
- Triggering alerts when drift exceeds threshold
- Automatically retraining the model
- Serving predictions via API
- Providing a monitoring dashboard

---

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Evidently AI (Drift Detection)
- Streamlit (Dashboard)
- FastAPI (Model Deployment)
- Joblib (Model Saving)

---

## Features

### 1. Data Drift Detection
- Uses **Evidently AI** for statistical drift detection (local environment)
- Custom drift logic used in Streamlit (cloud compatibility)

---

### 2. Drift Alert System
- Detects when drift exceeds threshold  
- Displays alerts in dashboard  

---

### 3. Automated Model Retraining
- Retrains model when drift is detected  
- Uses Scikit-learn pipeline (Scaler + Logistic Regression)  
- Saves updated model automatically  

---

### 4. Model Persistence
- Models stored using Joblib  
- Latest model always available for inference  

---

### 5. Interactive Dashboard (Streamlit)
- Drift simulation  
- Drift metrics visualization  
- Retrain trigger button  
- Real-time system feedback  

---

###  6. FastAPI Deployment
- REST API for predictions  
- JSON input → prediction output  
- Integrated with trained model  

---

##  System Workflow

```text
Incoming Data
      ↓
Drift Detection (Evidently / Custom)
      ↓
Drift Alert 🚨
      ↓
Model Retraining 🔄
      ↓
Updated Model Saved 💾
      ↓
Prediction via FastAPI ⚡
      ↓
Monitoring via Streamlit 📊

---

## How to Run

1 Clone Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


---
2 Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3. pip install -r requirements.txt

4. run streamlit dashboard
streamlit run app.py

5. run Fast API
uvicorn api:app --reload

6.Test API
http://127.0.0.1:8000/docs

Example API input

{
  "features": [0, -1.3, 0.2, 1.5, 0.3, -0.2, 0.1, 0.05, -0.01, 0.02, 0.03, -0.1, 0.2, 0.1, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 1.4, -1.5, 1.6, -1.7, 100]
}


ey Learnings
Real-world ML monitoring systems
Handling data drift in production
Integrating ML with APIs (FastAPI)
Building interactive dashboards
Managing model lifecycle (train → save → deploy)


Use Case

This project simulates how companies:

Monitor deployed ML models
Detect performance degradation
Automatically retrain models
Serve predictions via APIs

Author

Inderjeet Kaur

If you like this project

Give it a star ⭐ on GitHub!

