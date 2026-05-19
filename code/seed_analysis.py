import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 123, 456, 789, 1337] 

def load_data():
    df = pd.read_csv(r"C:\Users\mukun\model\ModelGuard\data\creditcard.csv")
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def noise_drift(X, sigma=1.0):
    return X + np.random.normal(0, sigma, X.shape)

def entropy(probs):
    probs = np.clip(probs, 1e-15, 1)
    return -np.sum(probs * np.log(probs), axis=1).mean()

def reliability_score(acc_delta, conf_delta, ent_delta):
    return (abs(acc_delta) + abs(conf_delta) + abs(ent_delta)) / 3

results = []

for seed in SEEDS:
    print(f"Running seed: {seed}")
    np.random.seed(seed)
    
    X, y = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    models = {
        'LR': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed),
        'RF': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed),
        'XGB': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            scale_pos_weight=len(y_train)/sum(y_train), random_state=seed)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Baseline (no drift)
        probs_base = model.predict_proba(X_test)
        acc_base = (model.predict(X_test) == y_test).mean()
        conf_base = probs_base.max(axis=1).mean()
        ent_base = entropy(probs_base)
        
        # Noise drift σ=1.0
        X_test_drift = noise_drift(X_test, sigma=1.0)
        probs_drift = model.predict_proba(X_test_drift)
        acc_drift = (model.predict(X_test_drift) == y_test).mean()
        conf_drift = probs_drift.max(axis=1).mean()
        ent_drift = entropy(probs_drift)
        
        r_score = reliability_score(acc_drift - acc_base, conf_drift - conf_base, ent_drift - ent_base)
        
        results.append({
            'seed': seed,
            'model': name,
            'acc_base': acc_base,
            'acc_drift': acc_drift,
            'conf_base': conf_base,
            'conf_drift': conf_drift,
            'ent_base': ent_base,
            'ent_drift': ent_drift,
            'reliability_score': r_score
        })

df_results = pd.DataFrame(results)
summary = df_results.groupby('model')['reliability_score'].agg(['mean', 'std', 'min', 'max']).round(5)
print("Reliability Score Summary (5 seeds)")
print(summary)

# Save
df_results.to_csv(r"C:\Users\mukun\model\ModelGuard\data\seed_analysis_results.csv", index=False)
summary.to_csv(r"C:\Users\mukun\model\ModelGuard\data\reliability_summary_seeds.csv")