# Capítulo 8 — Como Escolher o Algoritmo Certo

Não existe "melhor algoritmo universal". A escolha depende do problema, dos dados e das restrições práticas.

---

## Árvore de Decisão de Escolha

```
Seu dataset tem rótulos (labels)?
├── NÃO → Aprendizado não supervisionado (clustering, PCA) — fora do escopo
└── SIM → Aprendizado supervisionado
         │
         ├── Você quer prever um número contínuo? → Regressão (não abordado aqui)
         └── Você quer classificar em categorias?
                  │
                  ├── Quantas amostras?
                  │    ├── < 1.000 → KNN, SVM, Naive Bayes
                  │    ├── 1.000 – 100.000 → Random Forest, XGBoost, LightGBM
                  │    └── > 100.000 → XGBoost, LightGBM, Redes Neurais
                  │
                  ├── Você precisa entender POR QUÊ o modelo decidiu?
                  │    ├── SIM → Logistic Regression, Árvore de Decisão, SHAP + XGBoost
                  │    └── NÃO → Qualquer modelo
                  │
                  ├── As classes estão desbalanceadas?
                  │    ├── SIM → XGBoost (scale_pos_weight), + SMOTE/CTGAN
                  │    └── NÃO → Todos os modelos funcionam bem
                  │
                  └── Restrição de latência de inferência?
                       ├── < 1ms por amostra → Logistic Regression, Árvore única, XGBoost*
                       ├── < 100ms → Todos exceto KNN em datasets grandes
                       └── Sem restrição → Todos
```

*XGBoost com 1000 árvores e 7M amostras: ~14µs/amostra — dentro do requisito.

---

## Cheat Sheet Rápido

| Situação | Recomendação | Por quê |
|----------|-------------|---------|
| Precisa de uma baseline rápida | Logistic Regression | Treina em segundos, interpretável |
| Máxima precisão, dados tabulares | **XGBoost ou LightGBM** | Estado da arte em tabular |
| Dataset com muito ruído | Random Forest | Menos sensível a outliers |
| Precisa interpretar cada decisão | Árvore de Decisão (rasa) | Completamente visualizável |
| Features muito independentes | Naive Bayes | A premissa é válida |
| Precisa de anomalia sem rótulos | Isolation Forest, Autoencoder | Não precisam de labels |
| Poucos dados rotulados | KNN, SVM com kernel RBF | Não assumem forma da fronteira |

---

## Por que XGBoost foi a escolha certa para o artigo?

Checklist do nosso problema:

| Requisito | XGBoost atende? |
|-----------|----------------|
| ~14M amostras tabulares | ✓ escala bem |
| Classes desbalanceadas | ✓ `scale_pos_weight` |
| Features com correlações não lineares | ✓ árvores capturam não linearidades |
| Latência < 1ms em inferência | ✓ 14µs/amostra medido |
| Comparação com literatura | ✓ padrão em IDS automotivo |
| Importância de features | ✓ nativo |
| Robustez a overfitting | ✓ regularização L1/L2 + `gamma` |

---

## Como avaliar se você escolheu certo

### Passo 1: Baseline simples primeiro

Sempre comece com Logistic Regression ou Árvore rasa. Se eles já atingem 95% do desempenho, não precisa de algo mais complexo.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

baseline = LogisticRegression(random_state=42)
baseline.fit(X_train, y_train)
f1_base = f1_score(y_test, baseline.predict(X_test))
print(f"Baseline LR: F1={f1_base:.3f}")

# Agora tente XGBoost
# Se F1_xgb - F1_base < 0.02, a complexidade extra pode não valer
```

### Passo 2: Validação cruzada

Em vez de um único split treino/teste, usa K splits diferentes e tira a média.

```python
from sklearn.model_selection import cross_val_score
import xgboost as xgb

modelo = xgb.XGBClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(modelo, X, y, cv=5, scoring="f1", n_jobs=-1)
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

Se o desvio padrão é grande (ex: ±0.05), o modelo é instável — tente Random Forest.

### Passo 3: Curva de aprendizado

Mostra se você precisa de mais dados ou de um modelo melhor.

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    modelo, X, y, cv=5, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Treino")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validação")
plt.xlabel("Tamanho do conjunto de treino")
plt.ylabel("F1")
plt.legend()
plt.title("Curva de Aprendizado")
plt.show()
```

**Interpretando:**
```
Treino alto, Validação baixa e ambos estabilizados:
  → Overfitting → regularize mais ou use modelo mais simples

Treino e Validação baixos e próximos:
  → Underfitting → use modelo mais complexo ou mais features

Treino e Validação convergem conforme dados aumentam:
  → Mais dados ajudam → coletar mais dados

Treino ≈ Validação e ambos altos:
  → Modelo ideal para os dados atuais ✓
```

---

## Otimização de Hiperparâmetros

Quando você escolheu o algoritmo mas quer extrair o máximo:

### Grid Search (busca exaustiva)
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth":      [3, 6, 9],
    "learning_rate":  [0.01, 0.05, 0.1],
    "n_estimators":   [100, 500, 1000],
}

gs = GridSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)
gs.fit(X_train, y_train)
print(f"Melhor F1: {gs.best_score_:.4f}")
print(f"Melhores params: {gs.best_params_}")
```

### Optuna (busca bayesiana — mais eficiente)
```python
import optuna

def objective(trial):
    params = {
        "max_depth":      trial.suggest_int("max_depth", 3, 10),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":   trial.suggest_int("n_estimators", 100, 1000),
        "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
        "reg_lambda":     trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }
    modelo = xgb.XGBClassifier(**params, random_state=42)
    scores = cross_val_score(modelo, X_train, y_train, cv=3, scoring="f1")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"Melhor F1: {study.best_value:.4f}")
print(f"Melhores params: {study.best_params}")
```

---

## Resumo Final — Mapa Mental

```
                    DADOS TABULARES
                          │
              ┌───────────┴───────────┐
          Interpretável?           Máxima precisão?
              │                         │
         LR ou DT               XGBoost / LightGBM
              │                         │
         Simples, rápido         Regularização + CTGAN
         bom baseline            para desbalanceamento
              │                         │
              └───────────┬─────────────┘
                          │
                   Avaliar com:
                   F1 + ROC-AUC + PR-AUC
                   (não só acurácia!)
                          │
                   Comparar cenários:
                   Imbalanceado (realista)
                   Balanceado (benchmark)
```

---

## O que estudar a seguir

Para aprofundar além deste material:

| Tópico | Recurso sugerido |
|--------|-----------------|
| XGBoost original | Chen & Guestrin (2016) — "XGBoost: A Scalable Tree Boosting System" |
| Interpretabilidade | SHAP library — `pip install shap` |
| Otimização de hiperparâmetros | Optuna documentation |
| Dados desbalanceados | imbalanced-learn library |
| Aprendizado profundo para séries temporais | PyTorch + LSTM |
| IDS automotivo | Kim et al. (2026) — o artigo que estamos reproduzindo |
