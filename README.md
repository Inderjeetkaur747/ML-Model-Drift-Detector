#  ML Monitoring System with Data Drift Detection & Auto-Retraining

##  Overview
This project is an end-to-end Machine Learning system designed to monitor model performance in production by detecting data drift and automatically triggering model retraining.

It simulates a real-world fraud detection pipeline where incoming data distributions may change over time, affecting model accuracy.

---

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

1 Data Drift Detection using Evidently AI  
2 Drift Threshold Alert System  
3 Automatic Model Retraining  
4 Model Persistence (Saved Models)  
5 Interactive Dashboard (Streamlit)  
6 REST API for Predictions (FastAPI)  
7 Real-time Testing using Swagger UI  

---

## 🏗️ Project Architecture

Incoming Data
↓
Drift Detection (Evidently)
↓
Drift > Threshold?
↓ YES
Retrain Model
↓
Save Model
↓
Serve via FastAPI
↓
Streamlit Dashboard


---

## 📊 Dashboard Preview

- Drift Ratio Visualization  
- Drifted Columns Count  
- Drift Alert System  
- Retrain Model Button  

---

## How to Run

### Clone Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


---
### Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
