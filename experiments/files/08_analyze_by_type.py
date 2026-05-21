"""
Análise por tipo de ataque — pós 07_train_xgboost.py
Carrega o modelo treinado, prediz no teste e mostra recall/precision por attack_type.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import time

NPY_DIR = Path(r'C:\Mestrado\SDV_Research\data\npy')
CHUNK   = 500_000

FEAT_COLS = [
    'f01_ip_time_interval', 'f02_someip_likelihood', 'f03_someipsd_likelihood',
    'f04_tcpudp_likelihood', 'f05_someip_entropy', 'f06_someipsd_entropy',
    'f07_tcpudp_entropy', 'f08_someip_payload_changes', 'f09_someipsd_payload_changes',
    'f10_tcpudp_payload_changes', 'f11_ip_length_changes', 'f12_tcpudp_length_changes',
    'f13_payload_repeat_rate', 'f14_duplicate_source',
    'f15_someip_payload_length', 'f16_tcpudp_payload_length',
]

print('Carregando npy...')
X_train = np.load(NPY_DIR / 'X_train.npy')
y_train = np.load(NPY_DIR / 'y_train.npy').astype(int)
X_test  = np.load(NPY_DIR / 'X_test.npy')
y_test  = np.load(NPY_DIR / 'y_test.npy').astype(int)

print('Treinando modelo binario...')
t0 = time.time()
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.3,
    subsample=1.0, colsample_bytree=1.0, eval_metric='logloss',
    random_state=42, n_jobs=-1, tree_method='hist',
)
model.fit(X_train, y_train, verbose=False)
print(f'  Treino em {time.time()-t0:.1f}s')

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print(f'\n{"="*55}')
print(f'  BINÁRIO — 14 features (f14 cross-PCAP fix)')
print(f'{"="*55}')
print(f'  Acuracia  : {acc:.4f}')
print(f'  Precision : {prec:.4f}')
print(f'  Recall    : {rec:.4f}')
print(f'  F1-Score  : {f1:.4f}')
print(f'  AUC-ROC   : {auc:.4f}')
print(f'  TN={cm[0,0]:>9,}  FP={cm[0,1]:>7,}')
print(f'  FN={cm[1,0]:>9,}  TP={cm[1,1]:>7,}')

# ── Análise por tipo de ataque ──────────────────────────────────────────────
print('\nCarregando attack_type do test_features.csv...')
attack_col = []
for chunk in pd.read_csv(NPY_DIR / 'test_features.csv', usecols=['attack_type'], chunksize=CHUNK):
    attack_col.append(chunk['attack_type'].values)
attack_type = np.concatenate(attack_col)

print(f'  Distribuição no teste:')
for t, c in sorted(pd.Series(attack_type).value_counts().items()):
    print(f'    {t:<20} {c:>9,}')

print(f'\n{"="*70}')
print(f'  RECALL POR TIPO DE ATAQUE (quantos o modelo detecta)')
print(f'{"="*70}')
print(f'  {"tipo":<20}  {"total":>8}  {"TP":>8}  {"FN":>8}  {"recall":>8}  {"f14=1 (%)":>10}')
print(f'  {"-"*70}')

for at in sorted(set(attack_type)):
    mask = (attack_type == at)
    y_true_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    f14_sub    = X_test[mask, 13]   # f14_duplicate_source é coluna 13

    if at == 'normal':
        n_total = mask.sum()
        n_fp    = (y_pred_sub == 1).sum()
        print(f'  {"normal":<20}  {n_total:>8,}  {"—":>8}  {"—":>8}  {"—":>8}  f14=1: {100*f14_sub.mean():.2f}%  FP={n_fp:,}')
    else:
        n_total  = mask.sum()
        n_tp     = ((y_true_sub == 1) & (y_pred_sub == 1)).sum()
        n_fn     = ((y_true_sub == 1) & (y_pred_sub == 0)).sum()
        recall_t = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f14_rate = f14_sub.mean()   # fração dos pacotes desse tipo com f14=1 (normalizado [0,1])
        print(f'  {at:<20}  {n_total:>8,}  {n_tp:>8,}  {n_fn:>8,}  {recall_t:>8.4f}  f14=1: {100*f14_rate:.2f}%')

# ── Importância das features ─────────────────────────────────────────────────
print(f'\n  Importancia das features:')
feat_names = FEAT_COLS
for name, imp in sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1]):
    bar = '#' * int(imp * 40)
    print(f'  {name:<35}  {imp:.4f}  {bar}')

# ── f14 stats por tipo ────────────────────────────────────────────────────────
print(f'\n  f14_duplicate_source — distribuição por attack_type (pre-normalizacao: 0 ou 1):')
print(f'  {"tipo":<20}  {"f14=1 count":>12}  {"total":>10}  {"frac":>8}')
for at in sorted(set(attack_type)):
    mask    = (attack_type == at)
    f14_raw = X_test[mask, 13]
    n_ones  = (f14_raw > 0.5).sum()
    total   = mask.sum()
    print(f'  {at:<20}  {n_ones:>12,}  {total:>10,}  {n_ones/total:>8.4f}')
