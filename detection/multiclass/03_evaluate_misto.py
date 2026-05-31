"""
Cenário A — Avalia o modelo treinado no dataset sequencial sobre o dataset misto.

Responde: o modelo que aprendeu com capturas isoladas consegue detectar
ataques quando benigno e ataque coexistem no mesmo fluxo temporal?

Uso:
    python multiclass/03_evaluate_misto.py
    python multiclass/03_evaluate_misto.py --csv data/features_misto.csv
"""
import sys, time, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

ROOT      = Path(__file__).parent.parent
MODEL_DIR = Path(__file__).parent / 'model'
DATA_DIR  = Path(__file__).parent / 'data'

CLASS_NAMES = ['Benigno', 'DoS', 'Fuzzy', 'MITM_Multi', 'MITM_Single', 'FakeClientID']
FEAT_COLS   = [
    'f01_ip_time_interval',     'f08_someip_payload_change',
    'f11_ip_length_change',     'f12_tcpudp_length_change',
    'f13_payload_repeat_rate',  'f15_someip_payload_len',
    'f16_tcpudp_len',           'f17_src_packet_rate',
    'f18_src_payload_diversity','f19_is_sd',
    'f20_src_service_diversity','f21_is_relay_service',
    'f22_src_clientid_diversity',
]
CHUNK = 500_000


def load_model(model_dir: Path):
    path = model_dir / 'multiclass_classifier.json'
    if not path.exists():
        raise FileNotFoundError(f'Modelo não encontrado: {path}')
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def load_norm(model_dir: Path) -> dict:
    path = model_dir / 'norm_params.json'
    if not path.exists():
        raise FileNotFoundError(f'norm_params.json não encontrado: {path}')
    with open(path) as f:
        return json.load(f)


def normalize(X: np.ndarray, norm: dict) -> np.ndarray:
    X = X.copy().astype(np.float32)
    for j, col in enumerate(FEAT_COLS):
        if col in norm:
            lo = norm[col]['min']
            hi = norm[col]['max']
            d  = hi - lo
            X[:, j] = np.clip((X[:, j] - lo) / d, 0.0, 1.0) if d > 0 else 0.0
    return X


def evaluate(csv_path: Path, model, norm: dict) -> dict:
    print(f'\nCarregando {csv_path.name}...')
    x_chunks, y_chunks = [], []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK):
        if 'f22_src_clientid_diversity' not in chunk.columns:
            chunk['f22_src_clientid_diversity'] = 1.0
        x_chunks.append(chunk[FEAT_COLS].fillna(0).values)
        y_chunks.append(chunk['label'].values.astype(int))

    X = np.vstack(x_chunks)
    y = np.concatenate(y_chunks)
    print(f'  {len(X):,} amostras carregadas.')

    print('Distribuição real:')
    for i, name in enumerate(CLASS_NAMES):
        n = int((y == i).sum())
        if n > 0:
            print(f'  {name:<15}: {n:>10,}  ({n/len(y)*100:.1f}%)')

    print('\nNormalizando...')
    x_norm = normalize(X, norm)

    print('Classificando...')
    t0     = time.perf_counter()
    y_pred = model.predict(x_norm)
    t_ms   = (time.perf_counter() - t0) / len(X) * 1000

    t0_b       = time.perf_counter()
    model.predict_proba(x_norm)
    throughput = len(X) / (time.perf_counter() - t0_b)

    acc      = float(accuracy_score(y, y_pred))
    f1_mac   = float(f1_score(y, y_pred, average='macro',    zero_division=0))
    f1_wgt   = float(f1_score(y, y_pred, average='weighted', zero_division=0))
    f1_pc    = f1_score(y, y_pred, average=None, zero_division=0)
    cm       = confusion_matrix(y, y_pred, labels=list(range(6)))

    print('\n' + '='*60)
    print('  RESULTADO — Cenário A (modelo original × dataset misto)')
    print('='*60)
    print(f'  Acurácia   : {acc:.4f}')
    print(f'  F1 macro   : {f1_mac:.4f}')
    print(f'  F1 weighted: {f1_wgt:.4f}')
    print(f'  Lat./pacote: {t_ms:.3f} ms')
    print(f'  Throughput : {throughput:,.0f} pkt/s')
    print()
    print(classification_report(y, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return {
        'dataset':      csv_path.name,
        'n_samples':    int(len(X)),
        'accuracy':     acc,
        'f1_macro':     f1_mac,
        'f1_weighted':  f1_wgt,
        't_single_ms':  float(t_ms),
        'throughput_pkt_s': float(throughput),
        'f1_per_class': {
            CLASS_NAMES[i]: float(f1_pc[i]) for i in range(len(CLASS_NAMES))
        },
        'class_counts': {
            CLASS_NAMES[i]: int((y == i).sum()) for i in range(len(CLASS_NAMES))
        },
        'confusion_matrix': cm.tolist(),
    }


def main(csv_path: Path):
    for path, label in [(MODEL_DIR / 'multiclass_classifier.json', 'modelo'),
                        (MODEL_DIR / 'norm_params.json',           'norm_params'),
                        (csv_path,                                  'dataset misto')]:
        if not path.exists():
            print(f'ERRO: {label} não encontrado: {path}')
            print('Execute os passos anteriores:')
            print('  1. multiclass/02_train.py       (treino do modelo)')
            print('  2. multiclass/01c_mix_features.py (dataset misto)')
            sys.exit(1)

    model = load_model(MODEL_DIR)
    norm  = load_norm(MODEL_DIR)

    results = evaluate(csv_path, model, norm)

    out = MODEL_DIR / 'results_misto.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResultados salvos em {out}')

    # Comparação rápida com resultados originais
    orig_path = MODEL_DIR / 'results.json'
    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)
        print('\n-- Comparação: original vs misto ----------------------')
        print(f'  {"Métrica":<20} {"Original":>12} {"Misto":>12} {"Delta":>10}')
        print(f'  {"-"*56}')
        for key, label in [('f1_macro', 'F1 Macro'),
                            ('f1_weighted', 'F1 Weighted'),
                            ('accuracy', 'Acurácia')]:
            orig_v = orig.get(key, 0)
            misto_v = results[key]
            delta = misto_v - orig_v
            print(f'  {label:<20} {orig_v:>12.4f} {misto_v:>12.4f} {delta:>+10.4f}')

        print(f'\n  {"Classe":<20} {"F1 orig":>10} {"F1 misto":>10} {"Delta":>10}')
        print(f'  {"-"*52}')
        for cls in CLASS_NAMES:
            ov = orig.get('f1_per_class', {}).get(cls, None)
            mv = results['f1_per_class'].get(cls, 0)
            if ov is not None:
                print(f'  {cls:<20} {ov:>10.4f} {mv:>10.4f} {mv-ov:>+10.4f}')
            else:
                print(f'  {cls:<20} {"N/A":>10} {mv:>10.4f} {"N/A":>10}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(DATA_DIR / 'features_misto.csv'),
                        help='CSV do dataset misto (padrão: data/features_misto.csv)')
    args = parser.parse_args()
    main(Path(args.csv))
