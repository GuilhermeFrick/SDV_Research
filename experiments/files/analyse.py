import numpy as np
from pathlib import Path

npy_dir = Path(r"C:\Mestrado\SDV_Research\data\dataset_ism_xgboost\tr_te_sets")

X_train = np.load(npy_dir / "X_train.npy")
y_train = np.load(npy_dir / "y_train.npy")
X_test  = np.load(npy_dir / "X_test.npy")
y_test  = np.load(npy_dir / "y_test.npy")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test  shape:", X_test.shape)
print("y_test  shape:", y_test.shape)
print()
print("y_train valores únicos:", np.unique(y_train, return_counts=True))
print("y_test  valores únicos:", np.unique(y_test,  return_counts=True))
print()
print("X_train dtype:", X_train.dtype)
print("X_train — primeiras 3 linhas:")
print(X_train[:3])
print()
print("X_train — min/max por coluna:")
print("  min:", X_train.min(axis=0).round(4))
print("  max:", X_train.max(axis=0).round(4))

