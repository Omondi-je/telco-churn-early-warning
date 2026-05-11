#!/usr/bin/env python3
"""Production retraining script — callable by CI/CD or cron."""

import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import shutil
import os
import sys

def load_data(path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    return df.drop('customerID', axis=1)

def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'No': 0, 'Yes': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    categorical = [c for c in X.columns if c not in numeric]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])),
        'timestamp': datetime.now().isoformat()
    }
    
    return pipeline, metrics, X_test, y_test

def promote_if_better(new_pipeline, new_metrics, threshold=0.01):
    try:
        old_pipeline = joblib.load('models/phase4_logistic_regression.pkl')
        old_pred = old_pipeline.predict(X_test)
        old_recall = recall_score(y_test, old_pred)
    except FileNotFoundError:
        old_recall = 0.0
    
    if new_metrics['recall'] > old_recall + threshold:
        os.makedirs('models/archive', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy('models/phase4_logistic_regression.pkl', f'models/archive/phase4_{timestamp}.pkl')
        joblib.dump(new_pipeline, 'models/phase4_logistic_regression.pkl')
        
        with open('models/metrics_log.jsonl', 'a') as f:
            f.write(json.dumps(new_metrics) + '\n')
        
        print(f"PROMOTED: recall {old_recall:.4f} → {new_metrics['recall']:.4f}")
        return True
    else:
        print(f"REJECTED: recall {old_recall:.4f} vs new {new_metrics['recall']:.4f}")
        return False

if __name__ == '__main__':
    print(f"Retraining started: {datetime.now().isoformat()}")
    
    df = load_data()
    pipeline, metrics, X_test, y_test = train_model(df)
    
    print(f"New model metrics: {metrics}")
    
    promoted = promote_if_better(pipeline, metrics)
    sys.exit(0 if promoted else 0)  # Exit 0 either way, log tells the story