import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("ModelGuard - Silent Failure Analysis")

st.markdown("""
ModelGuard explores how machine learning models lose uncertainty awareness under different types of distribution shift.
The goal is not to maximize accuracy - but to understand **when models should no longer be trusted**.
""")

# -----------------------------
# Load Data
# -----------------------------
data_path = Path(__file__).resolve().parent / "results" / "final_results.csv"
df = pd.read_csv(data_path)
drift_types_path = Path(__file__).resolve().parent / "results" / "drift_types_final.csv"
drift_types = pd.read_csv(drift_types_path)
# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controls")

view = st.sidebar.radio(
    "View Mode",
    ["Reliability Ranking", "Drift Type Sensitivity", "Single Model Analysis"]
)

metric = st.sidebar.selectbox(
    "Metric",
    ["Accuracy", "Confidence", "Entropy"]
)

metric_map = {
    "Accuracy": "raw_acc",
    "Confidence": "raw_conf",
    "Entropy": "raw_ent"
}

metric_col = metric_map[metric]

# -----------------------------
# RELIABILITY RANKING VIEW
# -----------------------------
if view == "Reliability Ranking":

    st.subheader("Relative Model Reliability Ranking")

    scores = []

    for model in df["model"].unique():
        df_m = df[df["model"] == model].sort_values("drift")

        base = df_m.iloc[0]
        last = df_m.iloc[-1]

        da = abs(last["raw_acc"] - base["raw_acc"])
        dc = abs(last["raw_conf"] - base["raw_conf"])
        de = abs(last["raw_ent"] - base["raw_ent"])

        score = (da + dc + de) / 3
        scores.append([model, score])

    rank_df = pd.DataFrame(scores, columns=["Model", "Reliability_Score"])
    rank_df["Rank"] = rank_df["Reliability_Score"].rank(ascending=False)

    st.dataframe(rank_df.sort_values("Rank"))

    plt.figure(figsize=(6,4))
    plt.bar(rank_df["Model"], rank_df["Reliability_Score"])
    plt.title("ModelGuard Reliability Score (Higher = More Aware)")
    plt.ylabel("Sensitivity to Drift")
    plt.grid(axis="y", alpha=0.3)
    st.pyplot(plt)

    st.markdown("""
**Interpretation:**  
Higher scores indicate stronger reaction to drift.  
Lower scores indicate silent failure risk.
""")

# -----------------------------
# DRIFT TYPE SENSITIVITY
# -----------------------------
elif view == "Drift Type Sensitivity":

    st.subheader("Drift-Type Sensitivity Analysis")

    dtype = st.selectbox(
        "Select Drift Type",
        drift_types["drift_type"].unique()
    )

    df_t = drift_types[drift_types["drift_type"] == dtype]

    plt.figure(figsize=(8,5))

    for m in df_t["model"].unique():
        d = df_t[df_t["model"] == m].sort_values("drift")
        plt.plot(d["drift"], d[metric_col], marker="o", linewidth=2.5, label=m)

    plt.title(f"{metric} under {dtype} drift")
    plt.xlabel("Drift Intensity")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend()
    st.pyplot(plt)

    st.markdown("""
**Interpretation:**  
Different models fail silently under different drift patterns.
""")

# -----------------------------
# SINGLE MODEL VIEW
# -----------------------------
else:

    model = st.selectbox("Select Model", df["model"].unique())

    df_m = df[df["model"] == model].sort_values("drift")

    plt.figure(figsize=(8,5))
    plt.plot(df_m["drift"], df_m[metric_col], marker="o", linewidth=2.5)
    plt.title(f"{model} - {metric} under Drift")
    plt.xlabel("Drift Intensity")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    st.pyplot(plt)

    if model == "LR":
        st.info("Logistic Regression reacts strongly to drift - failure is visible.")
    elif model == "RF":
        st.warning("Random Forest partially reacts - silent failure risk.")
    else:
        st.error("XGBoost shows confidence rigidity - highest silent failure risk.")

# -----------------------------
# Closing Note
# -----------------------------
st.markdown("""
---
### ModelGuard Insight

High accuracy does not guarantee reliability.  
Models must not only perform well, they must know when they are no longer reliable.
""")
