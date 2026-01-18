import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# Title & Intro
# -----------------------------
st.title("Silent Failure in ML Models")

st.markdown("""
This dashboard explores how increasing model complexity improves accuracy while progressively suppressing uncertainty awareness, leading to silent failure under distribution shift in imbalanced datasets.
""")

# -----------------------------
# Load Data
# -----------------------------
data_path = Path(__file__).resolve().parent / "data" / "final_results.csv"
df = pd.read_csv(data_path)

# -----------------------------
# Controls
# -----------------------------
mode = st.radio("View Mode", ["Single Model", "Compare Models"], horizontal=True)

metric = st.selectbox(
    "Select Metric",
    ["Accuracy", "Confidence", "Entropy"]
)

metric_map = {
    "Accuracy": ("raw_acc","sig_acc","iso_acc","Accuracy"),
    "Confidence": ("raw_conf","sig_conf","iso_conf","Mean Confidence"),
    "Entropy": ("raw_ent","sig_ent","iso_ent","Mean Entropy")
}

raw_col, sig_col, iso_col, ylabel = metric_map[metric]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8,5))

if mode == "Single Model":
    model = st.selectbox("Select Model", df["model"].unique())
    mdf = df[df["model"] == model].sort_values("drift")

    plt.plot(mdf["drift"], mdf[raw_col], marker="o", linewidth=2.5, label="Raw")
    plt.plot(mdf["drift"], mdf[sig_col], marker="o", linewidth=2.5, linestyle="--", label="Sigmoid")
    plt.plot(mdf["drift"], mdf[iso_col], marker="o", linewidth=2.5, linestyle=":", label="Isotonic")

    plt.title(f"{model} — {ylabel} under Distribution Shift", fontsize=14, weight="bold")

else:
    for model in df["model"].unique():
        mdf = df[df["model"] == model].sort_values("drift")
        plt.plot(mdf["drift"], mdf[raw_col], marker="o", linewidth=2.5, label=model)

    plt.title(f"Raw {ylabel} Comparison Across Models", fontsize=14, weight="bold")

plt.xlabel("Drift Intensity", fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.grid(True, alpha=0.3)
plt.legend(frameon=False)

st.pyplot(plt)

# -----------------------------
# Interpretation Text
# -----------------------------
st.markdown("---")

if metric == "Accuracy":
    st.markdown("""
**Interpretation:**  
Accuracy remains high for complex models even under strong drift, masking reliability collapse in minority class detection.
""")

elif metric == "Confidence":
    st.markdown("""
**Interpretation:**  
Flat confidence curves indicate loss of uncertainty awareness. The model does not realize when it becomes unreliable.
""")

else:
    st.markdown("""
**Interpretation:**  
Entropy should increase under drift. Flat entropy means the model is blind to distribution shift.
""")

# -----------------------------
# Model-Specific Insight
# -----------------------------
if mode == "Single Model":
    if model == "LR":
        st.info("Logistic Regression shows visible uncertainty under drift — failure is detectable.")
    elif model == "RF":
        st.warning("Random Forest maintains confidence despite drift — silent failure begins.")
    else:
        st.error("XGBoost shows extreme confidence rigidity — silent failure is most severe.")

# -----------------------------
# Footer Message
# -----------------------------
st.markdown("""
---
**Key Insight:**  
Optimizing only for accuracy produces models that appear reliable while progressively losing their ability to signal uncertainty, resulting in silent reliability collapse.
""")
