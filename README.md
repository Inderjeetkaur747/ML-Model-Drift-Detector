#  ML Monitoring System with Data Drift Detection & Auto-Retraining

##  Overview
This project is an end-to-end Machine Learning system designed to monitor model performance in production by detecting data drift and automatically triggering model retraining.

It simulates a real-world fraud detection pipeline where incoming data distributions may change over time, affecting model accuracy.

---
**Live App:** https://ml-model-drift-detector-dkfqsnvvgqvndre9hrcbqz.streamlit.app/

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

## Project Architecture

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

## Dashboard Preview

- Drift Ratio Visualization  
- Drifted Columns Count  
- Drift Alert System  
- Retrain Model Button  

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


Key Learnings
Importance of monitoring ML models in production
Handling imbalanced datasets (fraud detection)
Implementing automated retraining pipelines
Building full-stack ML systems (UI + API)


Use Case

This project simulates how companies monitor production ML models to ensure:

Model reliability
Performance stability
Automated retraining pipelines


Author

Inderjeet Kaur

If you like this project

Give it a star ⭐ on GitHub!

