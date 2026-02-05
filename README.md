
#  ML Drift Detection System (End-to-End MLOps Project)

## Project Overview

This project demonstrates an end-to-end Machine Learning Drift Detection system with a production-style dashboard. The system monitors **data (feature) drift** between a baseline (training) dataset and incoming production data using **statistical hypothesis testing (Kolmogorov–Smirnov test)**.

The goal is to simulate **real-world ML monitoring**, where models may degrade over time due to changes in data distribution, even if accuracy initially looks fine.

This project goes beyond model training and focuses on **post-deployment monitoring**, a critical but often overlooked part of ML systems.

---

## Key Objectives

* Detect **feature-level data drift** between baseline and production data
* Generate **alerts** when drift exceeds a defined threshold
* Provide a **visual dashboard** for monitoring drift
* Simulate real-world **MLOps / production ML** workflows

---

## Concepts Covered

* Data Drift vs Model Drift
* Kolmogorov–Smirnov (KS) statistical test
* Feature-wise drift detection
* Threshold-based alerting
* ML monitoring dashboards
* Production-style ML pipelines

---

## Project Architecture

```
ML_Drift_Detector/
│
├── app.py                     # Streamlit UI (Dashboard)
├── README.md                 # Project documentation
│
├── data/
│   ├── baseline_data.csv      # Training / reference dataset
│   └── production_data.csv    # Incoming production dataset
│
├── src/
│   ├── drift_detection.py     # KS-test based drift logic
│   └── drift_reporting.py 
        model.py
        data_loader.py
        
      
│
└── screenshots/
        └── outputs       # UI screenshots
```

---

## Dataset Used

* **Breast Cancer Wisconsin Dataset**
* 30 numerical features describing tumor characteristics
* Baseline dataset represents **training-time data**
* Production dataset is a **modified version** simulating real-world drift

---

## Drift Detection Methodology

### Statistical Test Used: **Kolmogorov–Smirnov Test**

For each feature:

* Null Hypothesis (H₀): Baseline and production feature distributions are the same
* Alternative Hypothesis (H₁): Distributions are different

**Decision Rule:**

* If `p-value < 0.05` → Drift detected

---

### Drift Metrics Tracked

* KS Statistic (distance between distributions)
* p-value (statistical significance)
* Drifted features count
* Drift percentage

---

## Alerting Logic

The system triggers an alert when:

* One or more features show statistically significant drift

Example output:

```
Total Features: 30
Drifted Features: 5
Drift Percentage: 16.67%
ALERT: Significant feature drift detected!
```

This mimics **real production monitoring systems** used in industry.

---

## Streamlit Dashboard Features

* Upload baseline (training) data
* Upload production data
* Automatic drift computation
* Feature-wise drift table
* Clear alerts and summaries

The dashboard allows **non-technical stakeholders** to understand drift visually.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ML_Drift_Detector.git
cd ML_Drift_Detector
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

---

## 5 How Drift Was Simulated

To demonstrate drift:

* Selected numerical features were **scaled, shifted, or noise-injected**
* This simulates changes caused by:

  * Sensor drift
  * Data collection changes
  * Population behavior shifts

This ensures **real, statistically detectable drift**.

---

## Sample Output

* Feature drift table with KS statistics and p-values
* Alert banner when drift is detected
* Summary metrics showing drift percentage

(Screenshots can be added in the `screenshots/` folder)

---

## Future Improvements

* Add model retraining trigger
* Add prediction drift detection
* Integrate Evidently AI
* Deploy on cloud (AWS / GCP / Azure)
* Store drift logs in a database

---


check the link - https://ml-model-drift-detector-ng44rzwcx4uvzzrqxrqgjn.streamlit.app/

## Author

**Inderjeet Kaur**

Aspiring AI / ML Engineer | Python | Machine Learning | MLOps

---

If you found this project useful, feel free to star the repository!
