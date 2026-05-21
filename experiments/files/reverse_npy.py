#!/usr/bin/env python3
"""Reverse-engineering do dataset numpy publicado por Kim et al. (2026)."""
import numpy as np
import pandas as pd

NPY_DIR = r'C:\Mestrado\SDV_Research\data\dataset_ism_xgboost\dataset_ism_xgboost\tr_te_sets'

X_train = np.load(f'{NPY_DIR}\\X_train.npy', mmap_mode='r')
y_train = np.load(f'{NPY_DIR}\\y_train.npy')
X_test  = np.load(f'{NPY_DIR}\\X_test.npy',  mmap_mode='r')
y_test  = np.load(f'{NPY_DIR}\\y_test.npy')

X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

print(f"=== SHAPE ===")
print(f"X_train: {X_train.shape}  dtype={X_train.dtype}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}  classes={np.unique(y_train)}")
print(f"Total  : {X_all.shape[0]:,} amostras  |  normal={( y_all==0).sum():,}  ataque={(y_all==1).sum():,}")

print(f"\n=== ESTATÍSTICAS POR COLUNA (all data) ===")
print(f"{'col':<4} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'%zero':>8} {'%nan':>8}")
print("-" * 65)
for j in range(X_all.shape[1]):
    col = X_all[:, j]
    pct_zero = (col == 0).mean() * 100
    pct_nan  = np.isnan(col).mean() * 100
    print(f"  {j:<2}  {col.min():>10.4f} {col.max():>10.4f} {col.mean():>10.4f} "
          f"{col.std():>10.4f} {pct_zero:>7.1f}% {pct_nan:>7.1f}%")

print(f"\n=== RELAÇÕES MATEMÁTICAS (likelihood + entropy = 1?) ===")
pairs = [(0, 1, 'SOME/IP  col0+col1'), (2, 4, 'SOME/IP-SD col2+col4'), (3, 5, 'TCP/UDP  col3+col5')]
for a, b, label in pairs:
    s = X_all[:, a] + X_all[:, b]
    print(f"  {label}:  sum_min={s.min():.6f}  sum_max={s.max():.6f}  "
          f"sum_mean={s.mean():.6f}  exact_1={(np.abs(s-1)<1e-5).mean()*100:.2f}%")

print(f"\n=== ESTATÍSTICAS POR CLASSE ===")
COLS = ['col0','col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11']
FEAT_GUESS = [
    'SOME/IP likelihood',
    'SOME/IP entropy',
    'SOME/IP-SD likelihood',
    'TCP/UDP likelihood',
    'SOME/IP-SD entropy',
    'TCP/UDP entropy',
    'SOME/IP payload changes',
    '??? (mean=0.5)',
    'TCP/UDP payload changes',
    '??? (mean=0.005)',
    'IP length changes',
    'TCP/UDP length changes',
]
print(f"{'col':<4} {'feature':<28} {'normal_mean':>12} {'attack_mean':>12} {'delta':>10}")
print("-" * 75)
for j in range(12):
    n_mean = X_all[y_all==0, j].mean()
    a_mean = X_all[y_all==1, j].mean()
    delta  = a_mean - n_mean
    print(f"  {j:<2}  {FEAT_GUESS[j]:<28}  {n_mean:>12.4f}  {a_mean:>12.4f}  {delta:>+10.4f}")

print(f"\n=== CORRELAÇÃO ENTRE COLUNAS (|r| > 0.3) ===")
df_X = pd.DataFrame(X_all[:100_000], columns=[f'c{j}' for j in range(12)])
corr = df_X.corr().abs()
for i in range(12):
    for j in range(i+1, 12):
        r = corr.iloc[i, j]
        if r > 0.3:
            sign = '+' if df_X.iloc[:,i].corr(df_X.iloc[:,j]) > 0 else '-'
            print(f"  c{i} × c{j}  r={sign}{r:.3f}  ({FEAT_GUESS[i]}  ×  {FEAT_GUESS[j]})")

print(f"\n=== DISTRIBUIÇÃO POR CLASSE — col7 e col9 (as ambíguas) ===")
for j, name in [(7, 'col7'), (9, 'col9')]:
    for cls, cls_name in [(0, 'normal'), (1, 'ataque')]:
        vals = X_all[y_all==cls, j]
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        print(f"  {name} [{cls_name}]: mean={vals.mean():.4f}  "
              f"std={vals.std():.4f}  p5={p5:.4f}  p50={p50:.4f}  p95={p95:.4f}")

print(f"\n=== PERCENTIS COLUNA 7 (full) ===")
p = np.percentile(X_all[:, 7], [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
labels_p = ['0%','1%','5%','10%','25%','50%','75%','90%','95%','99%','100%']
for lp, pv in zip(labels_p, p):
    print(f"  {lp:>4}: {pv:.6f}")

print(f"\n=== VALORES ÚNICOS col7 (primeiros 20) ===")
uniq7 = np.sort(np.unique(np.round(X_all[:, 7], 4)))
print(f"  {uniq7[:20]}")
print(f"  total únicos: {len(np.unique(np.round(X_all[:, 7], 6))):,}")
