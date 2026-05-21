"""
Treinamento XGBoost — reprodução de Kim et al. (2026)
Carrega X_train/y_train, treina, avalia em X_test/y_test.
"""
import numpy as np
import time
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

NPY_DIR = Path(r'C:\Mestrado\SDV_Research\data\npy')

print('Carregando dados...')
X_train = np.load(NPY_DIR / 'X_train.npy')
y_train = np.load(NPY_DIR / 'y_train.npy').astype(int)
X_test  = np.load(NPY_DIR / 'X_test.npy')
y_test  = np.load(NPY_DIR / 'y_test.npy').astype(int)

print(f'  Treino: {X_train.shape}  ataque={y_train.sum():,} ({100*y_train.mean():.1f}%)')
print(f'  Teste : {X_test.shape}   ataque={y_test.sum():,} ({100*y_test.mean():.1f}%)')

# Hiperparâmetros de Kim et al. (2026) — Tabela 2 do artigo
# n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0
# (padrões do XGBoost que Kim provavelmente usou)
params = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    tree_method='hist',   # mais rápido para datasets grandes
)

print(f'\nTreinando XGBoost {params}...')
t0 = time.time()
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, verbose=False)
t_train = time.time() - t0
print(f'  Treino concluido em {t_train:.1f}s')

print('\nAvaliando no conjunto de teste...')
t0 = time.time()
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
t_inf = time.time() - t0

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print(f'\n{"="*55}')
print(f'  RESULTADOS — XGBoost no conjunto de teste')
print(f'{"="*55}')
print(f'  Acuracia  : {acc:.4f}  ({100*acc:.2f}%)')
print(f'  Precision : {prec:.4f}')
print(f'  Recall    : {rec:.4f}')
print(f'  F1-Score  : {f1:.4f}')
print(f'  AUC-ROC   : {auc:.4f}')
print(f'  Tempo inf : {t_inf:.2f}s')
print(f'\n  Matriz de confusao:')
print(f'    TN={cm[0,0]:>9,}  FP={cm[0,1]:>7,}')
print(f'    FN={cm[1,0]:>9,}  TP={cm[1,1]:>7,}')

print(f'\n  Referência Kim et al. (2026):')
print(f'    Accuracy : ~0.9900+')
print(f'    F1-Score : ~0.9800+')
print(f'    AUC-ROC  : ~0.9900+')

print(f'\n  Importancia das features:')
importances = model.feature_importances_
feat_names = [
    'f01_ip_time_interval','f02_someip_likelihood','f03_someipsd_likelihood',
    'f04_tcpudp_likelihood','f05_someip_entropy','f06_someipsd_entropy',
    'f07_tcpudp_entropy','f08_someip_payload_changes','f09_someipsd_payload_changes',
    'f10_tcpudp_payload_changes','f11_ip_length_changes','f12_tcpudp_length_changes',
    'f13_payload_repeat_rate','f14_duplicate_source',
]
for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
    bar = '#' * int(imp * 40)
    print(f'  {name:<35}  {imp:.4f}  {bar}')
