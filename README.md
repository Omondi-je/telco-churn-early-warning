# Telco Customer Churn Early Warning System

> **End-to-end MLOps pipeline** for predicting and preventing customer churn in telecommunications. Built with scikit-learn, SHAP, FastAPI, Docker, and GitHub Actions.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Business Problem](#business-problem)
- [Solution Architecture](#solution-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Phase Guide](#phase-guide)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Monitoring & Retraining](#monitoring--retraining)
- [Key Insights](#key-insights)
- [Future Work](#future-work)

---

## Business Problem

**Churn costs the telecom industry $10B+ annually.** Acquiring a new customer costs 5-25× more than retaining an existing one.

| Metric | Value |
|--------|-------|
| Dataset | 7,043 customers |
| Churn rate | 26.6% |
| Cost of false negative (missed churner) | ~$200-500 lifetime value |
| Cost of false positive (unnecessary retention spend) | ~$50-100 |

**Objective:** Identify at-risk customers *before* they churn, enabling proactive retention.

---

## Solution Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │───▶│   Train     │───▶│  Interpret  │───▶│   Deploy    │
│   Ingest    │    │   Model     │    │   (SHAP)    │    │   (FastAPI) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                        │
                              ┌─────────────┐          │
                              │   Monitor   │◀─────────┘
                              │   & Retrain │
                              └─────────────┘
```

**Stack:**
- **ML:** scikit-learn, SHAP, pandas
- **API:** FastAPI, Pydantic, Uvicorn
- **DevOps:** Docker, GitHub Actions, Makefile
- **Interpretability:** SHAP (SHapley Additive exPlanations)

---

## Project Structure

```
telco-churn-early-warning/
├── .github/
│   └── workflows/
│       └── retrain.yml           # Weekly automated retraining
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_baseline_model.ipynb   # Logistic Regression baseline
│   ├── 03_model_comparison.ipynb # LR vs RF vs Gradient Boosting
│   ├── 04_interpretation.ipynb   # SHAP + feature importance
│   ├── 05_deployment.ipynb       # Batch scoring + risk tiers
│   ├── 06_monitoring.ipynb       # Drift detection
│   ├── 07_retraining.ipynb       # Champion/challenger pipeline
│   └── 08_production.ipynb       # API testing & load tests
├── src/
│   ├── retrain.py                # Production retraining script
│   └── api.py                    # FastAPI prediction service
├── models/
│   ├── phase4_logistic_regression.pkl   # Production model
│   ├── archive/                  # Versioned model history
│   └── metrics_log.jsonl         # Audit trail
├── reports/
│   ├── figures/                  # SHAP plots, drift charts
│   └── critical_alerts.csv       # CRM-ready alert list
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Local orchestration
├── Makefile                      # Common commands
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
```

---

## Quick Start

### Prerequisites

```bash
python 3.12+
docker (optional)
```

### Install

```bash
git clone https://github.com/YOUR_USERNAME/telco-churn-early-warning.git
cd telco-churn-early-warning
pip install -r requirements.txt
```

### Run Notebook Pipeline

```bash
# Sequential execution
jupyter lab notebooks/
# Open 01_eda.ipynb → 02_baseline_model.ipynb → ... → 08_production.ipynb
```

### Run API Locally

```bash
# Docker (recommended)
docker-compose up --build

# Or native
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.0,"TotalCharges":140.0}'
```

### Retrain Model

```bash
make retrain
# Or manually
python src/retrain.py
```

---

## Phase Guide

| Phase | Notebook | Focus | Key Output |
|-------|----------|-------|------------|
| **1** | `01_eda.ipynb` | Understand data | Churn drivers identified (tenure, contract) |
| **2** | `02_baseline_model.ipynb` | Establish benchmark | 80.5% accuracy, 57.5% recall |
| **3** | `03_model_comparison.ipynb` | Select best model | Class-weighted LR wins (79.4% recall) |
| **4** | `04_interpretation.ipynb` | Explain predictions | SHAP plots + odds ratios |
| **5** | `05_deployment.ipynb` | Operationalize | Risk tiers + CRM alerts |
| **6** | `06_monitoring.ipynb` | Detect degradation | KS drift tests |
| **7** | `07_retraining.ipynb` | Auto-refresh | Champion/challenger pipeline |
| **8** | `08_production.ipynb` | Ship to production | Dockerized FastAPI |

---

## Model Performance

### Selected Model: Class-Weighted Logistic Regression

| Metric | Score | Notes |
|--------|-------|-------|
| **Recall (Churn)** | **79.4%** | Catches 4 of 5 churners |
| Accuracy | 72.6% | Lower by design (prioritizes recall) |
| ROC-AUC | 0.82 | Strong discrimination |
| False Negatives | 93 | 93 missed churners out of 1,409 test |

**Why recall over accuracy:** Missing a churner costs 5× more than a false alarm.

### Model Comparison

| Model | Accuracy | Recall | False Negatives |
|-------|----------|--------|-----------------|
| Logistic Regression (balanced) | 72.6% | **79.4%** | **93** ✅ |
| Random Forest | 76.1% | 74.2% | 112 |
| Gradient Boosting | 78.3% | 68.5% | 135 |

---

## API Reference

### `POST /predict`

Predict churn probability for a single customer.

**Request:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 140.0
}
```

**Response:**
```json
{
  "churn_probability": 0.8342,
  "risk_tier": "Critical",
  "will_churn": true
}
```

### `GET /health`

Health check for load balancers.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Deployment

### Docker

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

### Cloud (AWS Example)

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker tag telco-churn-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/telco-churn-api:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/telco-churn-api:latest

# Deploy to ECS/Fargate
aws ecs update-service --cluster telco-cluster --service churn-api --force-new-deployment
```

---

## Monitoring & Retraining

| Check | Method | Frequency | Trigger |
|-------|--------|-----------|---------|
| Data drift | Kolmogorov-Smirnov test | Weekly | p < 0.05 |
| Prediction drift | Distribution comparison | Weekly | Mean shift > 10% |
| Performance decay | Recall on holdout set | Weekly | Drop > 5% |
| Model refresh | Champion/challenger | Weekly | New recall > old + 1% |

**Automation:** GitHub Actions runs `make retrain` every Sunday 2 AM UTC. New models are auto-promoted if they outperform the incumbent.

---

## Key Insights

### 🔴 Top Churn Drivers (Push Factors)

| Feature | Odds Ratio | Interpretation |
|---------|-----------|----------------|
| Contract: Month-to-month | 4.2× | No commitment = highest risk |
| Internet: Fiber optic | 2.8× | Premium service, price-sensitive |
| Tenure: 0-12 months | 2.5× | New customers most volatile |
| Payment: Electronic check | 1.9× | Friction in payment process |
| No Tech Support | 1.7× | Poor service experience |

### 🟢 Retention Factors (Pull Factors)

| Feature | Odds Ratio | Interpretation |
|---------|-----------|----------------|
| Contract: Two year | 0.12× | Long commitment = sticky |
| Tenure: 48+ months | 0.18× | Loyalty compounds |
| Contract: One year | 0.35× | Medium commitment helps |
| Online Security: Yes | 0.62× | Value-add services retain |

### Business Rule: High-Risk Segment

```
IF tenure ≤ 12 months 
   AND Contract = Month-to-month 
   AND InternetService = Fiber optic
THEN churn risk = 67.4% (2.5× baseline)
```

**Action:** Proactive outreach with contract upgrade offer + tech support bundle.

---

## Future Work

| Enhancement | Impact | Effort |
|-------------|--------|--------|
| Real-time feature store | Sub-minute predictions | High |
| A/B test retention offers | Measure $ROI of interventions | Medium |
| Survival analysis | Predict *when* churn happens | Medium |
| NLP on support tickets | Sentiment as churn signal | High |
| Multi-model ensemble | +3-5% recall | Low |

---

## Author

Designed and engineered by Jeff Omondi Ooko

## License

MIT License — see [LICENSE](LICENSE) for details.
```

---

Save this as `README.md` in your repo root, then:

```bash
git add README.md
git commit -m "docs: valedictorian README with full project narrative

- Business problem framing with $ impact
- Solution architecture diagram
- Complete phase-by-phase guide
- API reference with request/response examples
- Deployment instructions (Docker + AWS)
- Monitoring & retraining summary table
- Key insights with odds ratios and business rules
- Future work roadmap"
git push origin main
```
