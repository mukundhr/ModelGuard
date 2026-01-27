# ModelGuard - Silent Failure Analysis in Machine Learning Models

**ModelGuard reveals when high-accuracy ML models become dangerously overconfident under distribution shift.**

---

## Overview

ModelGuard is a machine learning reliability analysis framework that studies **silent failure** - situations where models remain confident and well-performing while operating outside their training distribution.

Instead of asking:

> *How accurate is the model?*

ModelGuard asks:

> **Does the model know when it should not be trusted?**

The project evaluates Logistic Regression, Random Forest, and XGBoost on a highly imbalanced fraud dataset under multiple types of distribution shift, focusing on confidence, entropy, and reliability behavior rather than accuracy alone.

---

## Motivation

Traditional ML evaluation assumes that high accuracy and good calibration imply reliability.

However, in real-world deployment:

- Data distributions change.
- Minority classes degrade first.
- Models must still output predictions.
- Confidence often remains high even when reliability is lost.

This creates the most dangerous failure mode in ML systems:

> **Silent failure - confident predictions when the model no longer understands the data.**

ModelGuard was built to expose this phenomenon.

---

## Experimental Setup

### Dataset
Credit Card Fraud Detection Dataset (highly imbalanced)
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/dataset)

### Models
- Logistic Regression (LR)
- Random Forest (RF)
- XGBoost (XGB)

### Metrics
- Accuracy
- Mean Confidence
- Entropy (uncertainty awareness)
- Reliability Sensitivity Score

### Drift Types
- Noise drift (measurement noise)
- Scale drift (feature scaling mismatch)
- Bias drift (systematic shift)

Drift is injected directly into standardized feature space to simulate deployment-time distribution shift.

---

## Core Findings

### 1. Accuracy is not reliability

All models maintain high accuracy under drift.

However, confidence and entropy reveal that some models lose uncertainty awareness while appearing stable.

---

### 2. Model complexity inverts reliability

| Model | Accuracy | Uncertainty Awareness | Failure Behavior |
|------|---------|----------------------|-----------------|
| Logistic Regression | Lower | High | Failure is visible |
| Random Forest | High | Medium | Failure becomes silent |
| XGBoost | Highest | Lowest | Failure is most silent |

The most accurate model is the least aware of its own uncertainty.

---

### 3. Confidence rigidity is more dangerous than overfitting

The models are **not overfit** in the traditional sense.

They generalize well on IID test data.

However, under distribution shift they exhibit:

> **Confidence rigidity - remaining highly confident outside their training distribution.**

This is not a training failure.  
It is a deployment reliability failure.

---

### 4. Drift type matters

Different models fail silently under different drift patterns.

Bias drift is particularly dangerous:

- Logistic Regression reacts strongly.
- Random Forest reacts weakly.
- XGBoost remains almost perfectly confident.

This demonstrates that silent failure is both **model-dependent** and **drift-dependent**.

---
### Reliability Ranking
Logistic Regression > Random Forest > XGBoost


This ranking remains stable across multiple weighting schemes, demonstrating robustness.

---

## What Happens Beyond Ranking

Ranking alone is not the result.

The deeper result is:

> Performance ranking and reliability ranking are inverted.

XGBoost is the best performer - and the most dangerous under shift.

Logistic Regression is the weakest performer - and the most honest under shift.

---

## What ModelGuard Ultimately Shows

ModelGuard demonstrates that:

- High-capacity models suppress uncertainty.
- Suppressed uncertainty creates silent failure.
- Silent failure increases with model complexity.
- Reliability is a behavioral property, not a performance property.

---

## What ModelGuard Does NOT Claim

ModelGuard does not claim to eliminate silent failure.

Silent failure is an intrinsic consequence of generalization under uncertainty.

Instead, ModelGuard provides a framework to:

- Observe silent failure  
- Measure its severity  
- Compare models  
- Compare drift types  
- Understand reliability risk  

---

## Quantitative Summary

### Baseline Performance (No Drift)

| Model | Test Accuracy | Mean Confidence | Mean Entropy |
| --- | --- | --- | --- |
| Logistic Regression | ~97.6% | ~0.94 | ~0.15 |
| Random Forest | ~99.9% | ~0.99 | ~0.002 |
| XGBoost | ~99.95% | ~0.999 | ~0.0004 |

### Calibration Performance (ECE)

| Model | Raw ECE | Calibrated ECE |
| --- | --- | --- |
| LR | ~0.029 | ~0.0003 |
| RF | ~0.00020 | ~0.00017 |
| XGB | ~0.00032 | ~0.00013 |

### Reliability Sensitivity Score

| Model | Reliability Score |
| --- | --- |
| LR | ~0.081 |
| RF | ~0.009 |
| XGB | ~0.001 |

---

Run:

```bash
streamlit run app.py
```

