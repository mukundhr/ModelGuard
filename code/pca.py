import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\mukun\model\ModelGuard\data\creditcard.csv")
X = df.drop('Class', axis=1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def bias_drift(X, sigma=1.0):
    return X + sigma

X_test_base = X_test_scaled
X_test_bias = bias_drift(X_test_scaled, sigma=1.0)

pca = PCA(n_components=2, random_state=42)
X_pca_base = pca.fit_transform(X_test_base)
X_pca_bias = pca.transform(X_test_bias)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca_base[y_test == 0, 0], X_pca_base[y_test == 0, 1], 
            alpha=0.4, label='Legitimate', s=8, color='blue')
plt.scatter(X_pca_base[y_test == 1, 0], X_pca_base[y_test == 1, 1], 
            alpha=0.9, label='Fraud', s=25, color='red', marker='x')
plt.title('Baseline (σ = 0.0)', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_pca_bias[y_test == 0, 0], X_pca_bias[y_test == 0, 1], 
            alpha=0.4, label='Legitimate', s=8, color='blue')
plt.scatter(X_pca_bias[y_test == 1, 0], X_pca_bias[y_test == 1, 1], 
            alpha=0.9, label='Fraud', s=25, color='red', marker='x')
plt.title(f'Bias Drift (σ = {1.0})', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

plt.suptitle('PCA Projection Showing Increased Class Separation Under Bias Drift\n'
             '(Explains why Logistic Regression accuracy improves)', 
             fontsize=16, y=1.02)

plt.tight_layout()
plt.savefig('fig_bias_drift_pca.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'fig_bias_drift_pca.png'")