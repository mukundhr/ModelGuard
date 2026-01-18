# ModelGuard — Detecting Silent Failure in Machine Learning Models

**ModelGuard reveals when high-accuracy ML models become unreliable without realizing it.**

---

## Overview

ModelGuard is an interactive machine learning reliability analysis system that studies **silent failure** — situations where models remain confident and well-calibrated while becoming wrong under distribution shift.

Instead of asking *“How accurate is the model?”*, ModelGuard asks:

> **Does the model know when it should not be trusted?**

The project evaluates Logistic Regression, Random Forest, and XGBoost on a highly imbalanced fraud dataset using calibration, entropy, and drift simulations, and visualizes their reliability behavior through an interactive dashboard.

---

## Why ModelGuard Exists

Most ML projects optimize accuracy.

But in real systems:

- Accuracy can stay high while minority detection collapses  
- Confidence can remain extreme under drift  
- Calibration can look perfect while reliability is gone  
- Failures become invisible  

ModelGuard exposes **where that invisibility begins.**

---

## What ModelGuard Demonstrates

ModelGuard shows that:

> Increasing model complexity improves performance while progressively destroying uncertainty awareness.

Which means:

> Better models can fail more silently.

---

## Key Features

- Silent failure detection under distribution shift  
- Calibration comparison (Raw, Sigmoid, Isotonic)  
- Confidence and entropy based uncertainty analysis  
- Drift simulation on imbalanced data  
- Interactive Streamlit dashboard  
- Cross-model reliability comparison  

---

## Experimental Setup

### Dataset  
Credit Card Fraud Detection Dataset (highly imbalanced)

### Models  
- Logistic Regression  
- Random Forest  
- XGBoost  

### Calibration  
- Raw probabilities  
- Sigmoid calibration  
- Isotonic calibration  

### Metrics  
- Accuracy  
- Mean Confidence  
- Entropy  
- Expected Calibration Error (ECE)  

### Drift  
Gaussian covariate drift injected into standardized feature space.

---

## Core Findings

| Model | Accuracy | Uncertainty Awareness | Failure Behavior |
|------|---------|----------------------|-----------------|
| Logistic Regression | Lower | High | Failure is visible |
| Random Forest | High | Low | Failure becomes silent |
| XGBoost | Highest | Lowest | Failure is almost invisible |

### Key Insight

> As model complexity increases, failure becomes harder to detect.

Calibration improves probability alignment — **but not reliability awareness.**

---

## Interactive Dashboard

ModelGuard includes an interactive Streamlit dashboard to explore:

- Model behavior under drift  
- Confidence & entropy dynamics  
- Single model and comparison modes  
- Built-in interpretation  

Run locally:

```bash
streamlit run streamlit_app.py
